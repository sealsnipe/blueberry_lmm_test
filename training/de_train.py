"""
Training script for German PLASA LLM
Supports FP16/FP4 quantization, gradient accumulation, checkpointing, and logging
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import bitsandbytes for FP4 quantization
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("Warning: bitsandbytes not available. FP4 quantization disabled.")

# Import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Using local logging only.")

from configs import load_config, validate_config
from models import create_plasa_model
from data import GermanDatasetPreprocessor, DatasetConfig
from utils.logging import setup_logging
from utils.metrics import MetricsTracker


class TokenDataset(Dataset):
    """Dataset for token sequences"""
    def __init__(self, tokens: List[int], seq_len: int = 512, stride: int = None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        
        # Calculate number of samples
        self.num_samples = max(0, (len(tokens) - seq_len) // self.stride + 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x = torch.tensor(self.tokens[start_idx:start_idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:start_idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    precision: str = "fp16"
) -> Dict[str, float]:
    """
    Evaluate model on validation set
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        precision: Precision mode
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            if precision == "fp16":
                with autocast('cuda', dtype=torch.float16):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(min(avg_loss, 20))  # Clamp for numerical stability
    
    model.train()
    
    return {
        'val_loss': avg_loss,
        'val_perplexity': perplexity
    }


def create_optimizer(
    model: nn.Module,
    config: Dict,
    use_fp4: bool = False
):
    """
    Create optimizer with BitsAndBytes 8bit quantization (always used for VRAM optimization)
    
    Args:
        model: Model to optimize
        config: Training configuration
        use_fp4: Whether to use FP4 quantization (ignored, always use 8bit)
        
    Returns:
        Optimizer
    """
    training_config = config['training']
    
    # Always use BitsAndBytes 8bit optimizer for VRAM optimization
    if BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.1),
            betas=tuple(training_config.get('betas', [0.9, 0.95]))
        )
    else:
        # Fallback to standard AdamW if bitsandbytes not available
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.1),
            betas=tuple(training_config.get('betas', [0.9, 0.95]))
        )
    
    return optimizer


def create_scheduler(optimizer, config: Dict, num_steps: int):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        config: Training configuration
        num_steps: Total number of training steps
        
    Returns:
        Scheduler
    """
    training_config = config['training']
    scheduler_type = training_config.get('lr_scheduler', 'cosine')
    warmup_steps = training_config.get('warmup_steps', 500)
    min_lr = training_config.get('min_lr', 1e-6)
    max_lr = training_config['learning_rate']
    
    if scheduler_type == 'cosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr / max_lr + (1 - min_lr / max_lr) * cosine_decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Linear scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / max_lr,
            total_iters=num_steps
        )
    
    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    is_best: bool = False
):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current step
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Checkpoint directory
        is_best: Whether this is the best model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    # Save last checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer,
    scheduler
):
    """
    Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load into
        scheduler: Scheduler to load into
        
    Returns:
        step, epoch, metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    step = checkpoint.get('step', 0)
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return step, epoch, metrics


def train(
    config_path: str,
    device_id: int = 0,
    wandb_mode: str = "online",
    resume_from: Optional[str] = None,
    epochs_override: Optional[int] = None,
    subset_size_override: Optional[int] = None
):
    """
    Main training function
    
    Args:
        config_path: Path to config YAML file
        device_id: CUDA device ID
        wandb_mode: Wandb mode (online/offline/disabled)
        resume_from: Path to checkpoint to resume from
    """
    # Load configuration
    config = load_config(config_path)
    validate_config(config)
    
    # Override epochs if provided
    if epochs_override:
        config['training']['max_epochs'] = epochs_override
    
    # Override subset size if provided
    if subset_size_override:
        # Adjust dummy dataset size based on subset
        config['dataset']['preprocessing']['target_tokens'] = subset_size_override * 512
    
    # Setup logging
    logger = setup_logging(log_dir=config['logging']['log_dir'])
    logger.info(f"Starting training with config: {config_path}")
    
    # Setup device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup wandb
    if WANDB_AVAILABLE and config['logging']['wandb']['enabled']:
        wandb.init(
            project=config['logging']['wandb']['project'],
            entity=config['logging']['wandb'].get('entity'),
            mode=wandb_mode,
            config=config
        )
    
    # Create model
    logger.info("Creating model...")
    model = create_plasa_model(config)
    
    # Enable gradient checkpointing for VRAM optimization
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        # Fallback: Manual gradient checkpointing via torch.utils.checkpoint
        logger.info("Using manual gradient checkpointing")
    
    model = model.to(device)
    
    total_params = model.count_parameters()
    logger.info(f"Model created with {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # FP4 quantization (if enabled)
    precision = config['training']['precision']
    use_fp4 = config['training'].get('fp4_enabled', False) and precision == "fp4"
    
    if use_fp4:
        if not BNB_AVAILABLE:
            logger.warning("FP4 requested but bitsandbytes not available. Using FP16 instead.")
            precision = "fp16"
            use_fp4 = False
        else:
            logger.info("Using FP4 quantization")
            # Quantize model
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
    
    # Create optimizer
    optimizer = create_optimizer(model, config, use_fp4=use_fp4)
    
    # Load/prepare data
    logger.info("Loading data...")
    dataset_configs = []
    for ds_cfg in config['dataset']['datasets']:
        dataset_configs.append(DatasetConfig(
            name=ds_cfg['name'],
            path=ds_cfg['path'],
            weight=ds_cfg['weight'],
            streaming=ds_cfg.get('streaming', True),
            subset=ds_cfg.get('subset'),
            config=ds_cfg.get('config'),
            filters=ds_cfg.get('filters')
        ))
    
    # For now, use a simple tokenized dataset
    # In production, this would use the streaming preprocessor
    preprocessor = GermanDatasetPreprocessor(
        tokenizer_name="gpt2",
        min_length=config['dataset']['preprocessing']['min_length_tokens'],
        max_length=config['dataset']['preprocessing']['max_length_tokens'],
        target_length=config['dataset']['preprocessing']['target_length_tokens'],
        language_filter=config['dataset']['preprocessing']['language_filter']
    )
    
    # Create token dataset (simplified - in production use streaming)
    # For testing, we'll create a dummy dataset
    seq_len = config['dataset']['preprocessing']['target_length_tokens']
    vocab_size = config['model']['vocab_size']
    
    # Generate dummy tokens for testing (adjust size if subset_size provided)
    if subset_size_override:
        num_tokens = subset_size_override * 512  # Rough estimate
    else:
        num_tokens = 1000000
    dummy_tokens = torch.randint(0, vocab_size, (num_tokens,)).tolist()
    
    train_dataset = TokenDataset(dummy_tokens, seq_len=seq_len)
    val_dataset = TokenDataset(dummy_tokens[-100000:], seq_len=seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # VRAM optimization: no multiprocessing
        pin_memory=False  # VRAM optimization: disable pin memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # VRAM optimization: no multiprocessing
        pin_memory=False  # VRAM optimization: disable pin memory
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Calculate number of steps
    training_config = config['training']
    gradient_accumulation_steps = training_config['gradient_accumulation_steps']
    max_epochs = training_config.get('max_epochs', 3)
    
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    max_steps = training_config.get('max_steps')
    if max_steps is None:
        max_steps = steps_per_epoch * max_epochs
    
    logger.info(f"Training for {max_steps} steps ({max_epochs} epochs)")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, max_steps)
    
    # Setup mixed precision
    scaler = GradScaler() if precision == "fp16" and training_config.get('use_amp', True) else None
    
    # Resume from checkpoint
    step = 0
    epoch = 0
    best_val_loss = float('inf')
    
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        step, epoch, metrics = load_checkpoint(resume_from, model, optimizer, scheduler)
        best_val_loss = metrics.get('best_val_loss', float('inf'))
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config['logging']['metrics_file'])
    
    # VRAM tracking
    vram_peak = 0.0
    
    # Training loop
    model.train()
    logger.info("Starting training loop...")
    
    pbar = tqdm(total=max_steps, desc="Training")
    pbar.update(step)
    
    step_start_time = time.time()  # Per-step timer for TPS calculation
    epoch_start_time = time.time()
    
    while step < max_steps:
        epoch_start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass with autocast for VRAM optimization
            if precision == "fp16" and scaler is not None:
                with autocast('cuda', dtype=torch.float16):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                with autocast('cuda', dtype=torch.float16):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / gradient_accumulation_steps
                loss.backward()
            
            # Optimizer step
            if (step + 1) % gradient_accumulation_steps == 0:
                if precision == "fp16" and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        training_config.get('grad_clip', 1.0)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        training_config.get('grad_clip', 1.0)
                    )
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # VRAM monitoring every 10 steps
            if step % 10 == 0 and torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info(device)
                used_mem = total_mem - free_mem
                vram_gb = used_mem / 1e9
                vram_peak = max(vram_peak, vram_gb)
                print(f"Step {step}: VRAM {vram_gb:.2f}GB / {total_mem/1e9:.2f}GB (Peak: {vram_peak:.2f}GB)")
            
            # Logging
            if step % config['logging']['wandb'].get('log_freq', 10) == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_loss = loss.item() * gradient_accumulation_steps
                perplexity = math.exp(min(current_loss, 20))
                
                # Get VRAM usage
                if torch.cuda.is_available():
                    free_mem, total_mem = torch.cuda.mem_get_info(device)
                    used_mem = total_mem - free_mem
                    vram_used = used_mem / 1e9  # GB
                    vram_peak = max(vram_peak, vram_used)
                else:
                    vram_used = 0.0
                
                # Calculate tokens per second (per-step timer)
                elapsed_time = time.time() - step_start_time
                tokens_processed = x.numel() * gradient_accumulation_steps
                tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                step_start_time = time.time()  # Reset timer for next step
                
                metrics = {
                    'step': step,
                    'epoch': epoch,
                    'loss': current_loss,
                    'perplexity': perplexity,
                    'learning_rate': current_lr,
                    'vram_usage': vram_used,
                    'tokens_per_second': tokens_per_sec
                }
                
                # Save metrics
                metrics_tracker.log(metrics)
                
                # Log to wandb
                if WANDB_AVAILABLE and config['logging']['wandb']['enabled']:
                    wandb.log(metrics, step=step)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'ppl': f'{perplexity:.2f}',
                    'lr': f'{current_lr:.2e}',
                    'vram': f'{vram_used:.2f}GB'
                })
            
            # Evaluation
            if step > 0 and step % config['training']['eval_every'] == 0:
                logger.info(f"Evaluating at step {step}...")
                val_metrics = evaluate_model(model, val_loader, device, precision)
                
                is_best = val_metrics['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val PPL: {val_metrics['val_perplexity']:.2f}")
                
                if WANDB_AVAILABLE and config['logging']['wandb']['enabled']:
                    wandb.log({'val/' + k: v for k, v in val_metrics.items()}, step=step)
            
            # Checkpointing
            if step > 0 and step % config['training']['save_every'] == 0:
                checkpoint_metrics = {
                    'loss': loss.item() * gradient_accumulation_steps,
                    'best_val_loss': best_val_loss
                }
                save_checkpoint(
                    model, optimizer, scheduler, step, epoch,
                    checkpoint_metrics,
                    config['checkpoint']['dir'],
                    is_best=False
                )
            
            step += 1
            pbar.update(1)
        
        epoch += 1
        logger.info(f"Completed epoch {epoch}")
    
    pbar.close()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = evaluate_model(model, val_loader, device, precision)
    logger.info(f"Final Val Loss: {final_metrics['val_loss']:.4f}, "
                f"Final Val PPL: {final_metrics['val_perplexity']:.2f}")
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, step, epoch,
        {'final_val_loss': final_metrics['val_loss'], 'best_val_loss': best_val_loss},
        config['checkpoint']['dir'],
        is_best=final_metrics['val_loss'] == best_val_loss
    )
    
    # Finalize wandb
    if WANDB_AVAILABLE and config['logging']['wandb']['enabled']:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train German PLASA LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--wandb_mode", type=str, default="online", 
                       choices=["online", "offline", "disabled"],
                       help="Wandb mode")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--subset-size", type=int, default=None,
                       help="Dataset subset size for testing")
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        device_id=args.device,
        wandb_mode=args.wandb_mode,
        resume_from=args.resume_from,
        epochs_override=args.epochs,
        subset_size_override=args.subset_size
    )

