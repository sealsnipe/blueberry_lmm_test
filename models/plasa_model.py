"""
PLASA Attention Model for German Language Training
450M parameter model with DeepSeek Sparse Attention (PLASA)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Try to import sparse attention from temp_blueberry, or use inline implementation
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'temp_blueberry', 
                                    'experiments', 'exp2_in_progress_attention_mechanisms', 
                                    'exp1_sparse_vs_classic_attention'))
    from sparse_attention import DeepSeekSparseAttention
except ImportError:
    # Fallback: import from a local copy or implement inline
    # For now, we'll copy the key components inline
    print("Warning: Could not import sparse_attention. Using inline implementation.")
    
    # Inline implementation of DeepSeek Sparse Attention components
    from torchtune.modules import RotaryPositionalEmbeddings
    
    class LightningIndexer(nn.Module):
        """Lightning Indexer for DeepSeek Sparse Attention"""
        def __init__(self, d_model: int, indexer_heads: int = 4, indexer_dim: int = 64, dropout: float = 0.1):
            super().__init__()
            self.d_model = d_model
            self.indexer_heads = indexer_heads
            self.indexer_dim = indexer_dim
            self.q_proj = nn.Linear(d_model, indexer_heads * indexer_dim, bias=False)
            self.k_proj = nn.Linear(d_model, indexer_dim, bias=False)
            self.w_proj = nn.Linear(d_model, indexer_heads, bias=False)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape
            queries = self.q_proj(x).reshape(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
            keys = self.k_proj(x)
            weights = self.w_proj(x)
            dots = torch.einsum('bthd,bsd->bths', queries, keys)
            activated = F.relu(dots)
            weighted = activated * weights.unsqueeze(-1)
            index_scores = weighted.sum(dim=2)
            return index_scores
    
    class TopKTokenSelector(nn.Module):
        """Fine-grained Token Selection Mechanism"""
        def __init__(self, top_k: int = 512):
            super().__init__()
            self.top_k = top_k
        
        def forward(self, index_scores: torch.Tensor, apply_causal_mask: bool = True):
            batch_size, seq_len_q, seq_len_k = index_scores.shape
            if apply_causal_mask:
                causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=index_scores.device), diagonal=1).bool()
                index_scores = index_scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
            actual_k = min(self.top_k, seq_len_k)
            top_k_values, top_k_indices = torch.topk(index_scores, k=actual_k, dim=-1, largest=True)
            top_k_mask = torch.zeros_like(index_scores, dtype=torch.bool)
            top_k_mask.scatter_(2, top_k_indices, True)
            return top_k_mask, top_k_indices
    
    class DeepSeekSparseAttention(nn.Module):
        """DeepSeek Sparse Attention (DSA)"""
        def __init__(self, d_model: int, n_heads: int, max_seq_len: int, 
                     indexer_heads: int = 4, indexer_dim: int = 64, 
                     sparse_top_k: int = 512, dropout: float = 0.1):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.sparse_top_k = sparse_top_k
            self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
            self.w_o = nn.Linear(d_model, d_model, bias=False)
            self.rotary = RotaryPositionalEmbeddings(dim=self.d_k, max_seq_len=max_seq_len, base=10000)
            self.dropout = dropout
            self.indexer = LightningIndexer(d_model, indexer_heads, indexer_dim, dropout)
            self.selector = TopKTokenSelector(top_k=sparse_top_k)
            self.use_sparse = True
        
        def forward(self, x: torch.Tensor, return_index_scores: bool = False):
            batch_size, seq_len, _ = x.shape
            qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            Q, K, V = qkv[0], qkv[1], qkv[2]
            Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
            K = self.rotary(K.transpose(1, 2)).transpose(1, 2)
            index_scores = self.indexer(x)
            if self.use_sparse:
                top_k_mask, top_k_indices = self.selector(index_scores, apply_causal_mask=True)
                attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=x.device, dtype=Q.dtype)
                attn_mask = attn_mask.masked_fill(~top_k_mask.unsqueeze(1), float('-inf'))
                attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, 
                                                            dropout_p=self.dropout if self.training else 0.0)
            else:
                attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True, 
                                                            dropout_p=self.dropout if self.training else 0.0)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            output = self.w_o(attn_output)
            if return_index_scores:
                return output, index_scores
            return output, None
        
        def enable_sparse(self):
            self.use_sparse = True
        
        def disable_sparse(self):
            self.use_sparse = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))


class PLASATransformerBlock(nn.Module):
    """
    Transformer block with PLASA (DeepSeek Sparse) Attention
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # PLASA Attention
        self.attention = DeepSeekSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            sparse_top_k=sparse_top_k,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout)
        )
        
        # Normalization layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        return_index_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with layer-wise gradient checkpointing
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_index_scores: Whether to return index scores
            
        Returns:
            output: Block output
            index_scores: Index scores if requested
        """
        # Layer-wise checkpointing for VRAM optimization
        if self.training and hasattr(self, '_use_checkpoint') and self._use_checkpoint:
            import torch.utils.checkpoint as checkpoint
            
            # Checkpoint attention
            def attn_fn(x_norm):
                attn_out, idx_scores = self.attention(
                    x_norm,
                    return_index_scores=return_index_scores
                )
                return attn_out, idx_scores
            
            attn_out, index_scores = checkpoint.checkpoint(
                attn_fn, self.norm1(x), use_reentrant=False
            )
            x = x + attn_out
            
            # Checkpoint feed-forward
            def ff_fn(x_norm):
                return self.ff(x_norm)
            
            ff_out = checkpoint.checkpoint(
                ff_fn, self.norm2(x), use_reentrant=False
            )
            x = x + ff_out
        else:
            # Standard forward pass
            attn_out, index_scores = self.attention(
                self.norm1(x),
                return_index_scores=return_index_scores
            )
            x = x + attn_out
            
            # Feed-forward
            ff_out = self.ff(self.norm2(x))
            x = x + ff_out
        
        return x, index_scores


class PLASALLM(nn.Module):
    """
    450M Parameter Language Model with PLASA Attention
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 24,
        d_ff: int = 4096,
        max_seq_len: int = 1024,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        """
        Initialize PLASA LLM
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            indexer_heads: Number of indexer heads for PLASA
            indexer_dim: Indexer dimension for PLASA
            sparse_top_k: Top-k tokens for sparse attention
            dropout: Dropout probability
            tie_weights: Whether to tie embedding and output weights
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learned)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks with PLASA attention
        self.blocks = nn.ModuleList([
            PLASATransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                indexer_heads=indexer_heads,
                indexer_dim=indexer_dim,
                sparse_top_k=sparse_top_k,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing support
        self.gradient_checkpointing = False
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        self.gradient_checkpointing = True
        # Enable layer-wise checkpointing in blocks
        for block in self.blocks:
            block._use_checkpoint = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
        for block in self.blocks:
            block._use_checkpoint = False
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        return_index_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass
        
        Args:
            x: Input token indices [batch_size, seq_len]
            return_index_scores: Whether to return index scores from all layers
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            index_scores_list: List of index scores per layer (if requested)
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        x = self.embed(x)  # [batch_size, seq_len, d_model]
        
        # Positional embeddings - clamp positions to valid range
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = torch.clamp(positions, 0, self.pos_embed.weight.size(0) - 1)
        x = x + self.pos_embed(positions)
        
        # Pass through transformer blocks (with gradient checkpointing if enabled)
        index_scores_list = [] if return_index_scores else None
        
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for VRAM optimization
            import torch.utils.checkpoint as checkpoint
            for block in self.blocks:
                if return_index_scores:
                    x, index_scores = checkpoint.checkpoint(
                        block, x, return_index_scores, use_reentrant=False
                    )
                    index_scores_list.append(index_scores)
                else:
                    x, _ = checkpoint.checkpoint(
                        block, x, False, use_reentrant=False
                    )
        else:
            for block in self.blocks:
                x, index_scores = block(x, return_index_scores=return_index_scores)
                if return_index_scores and index_scores is not None:
                    index_scores_list.append(index_scores)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, index_scores_list
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def enable_sparse(self):
        """Enable sparse attention in all blocks"""
        for block in self.blocks:
            block.attention.enable_sparse()
    
    def disable_sparse(self):
        """Disable sparse attention (use dense) in all blocks"""
        for block in self.blocks:
            block.attention.disable_sparse()


def create_plasa_model(config: dict) -> PLASALLM:
    """
    Create PLASA model from configuration
    
    Args:
        config: Configuration dictionary with model settings
        
    Returns:
        PLASA model instance
    """
    model_config = config['model']
    attention_config = model_config.get('attention', {})
    
    model = PLASALLM(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        d_ff=model_config['d_ff'],
        max_seq_len=model_config['max_seq_len'],
        indexer_heads=attention_config.get('indexer_heads', 4),
        indexer_dim=attention_config.get('indexer_dim', 64),
        sparse_top_k=attention_config.get('sparse_top_k', 512),
        dropout=model_config.get('dropout', 0.1),
        tie_weights=True
    )
    
    return model

