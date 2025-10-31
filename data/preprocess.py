"""
German dataset preprocessing module
Handles multi-dataset loading, filtering, and tokenization
"""
import os
import json
from typing import List, Dict, Iterator, Optional
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import load_dataset, IterableDataset, Dataset
from transformers import AutoTokenizer
from langdetect import detect, LangDetectException


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    path: str
    weight: float
    streaming: bool = True
    subset: Optional[str] = None
    config: Optional[str] = None
    filters: Optional[List[str]] = None
    text_column: str = "text"


class GermanDatasetPreprocessor:
    """
    Preprocessor for German text datasets with streaming support
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        min_length: int = 256,
        max_length: int = 1024,
        target_length: int = 512,
        language_filter: str = "de",
        deduplication: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize preprocessor
        
        Args:
            tokenizer_name: HuggingFace tokenizer name
            min_length: Minimum sequence length in tokens
            max_length: Maximum sequence length in tokens
            target_length: Target sequence length for chunking
            language_filter: Language code to filter (e.g., "de")
            deduplication: Whether to deduplicate texts
            cache_dir: Cache directory for tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.min_length = min_length
        self.max_length = max_length
        self.target_length = target_length
        self.language_filter = language_filter
        self.deduplication = deduplication
        
        self.seen_texts = set() if deduplication else None
        
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect language of text
        
        Args:
            text: Input text
            
        Returns:
            Language code or None if detection fails
        """
        if not text or len(text.strip()) < 10:
            return None
        
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return None
    
    def filter_german(self, example: Dict) -> bool:
        """
        Filter examples to keep only German text
        
        Args:
            example: Dataset example with 'text' field
            
        Returns:
            True if text is German
        """
        text = example.get('text', '')
        
        # Basic length check
        if len(text.strip()) < 50:
            return False
        
        # Language detection
        if self.language_filter:
            lang = self.detect_language(text)
            if lang != self.language_filter:
                return False
        
        # Deduplication
        if self.deduplication and self.seen_texts is not None:
            text_hash = hash(text)
            if text_hash in self.seen_texts:
                return False
            self.seen_texts.add(text_hash)
        
        return True
    
    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens
    
    def chunk_tokens(self, tokens: List[int]) -> List[List[int]]:
        """
        Chunk tokens into sequences of target length
        
        Args:
            tokens: List of token IDs
            
        Returns:
            List of token chunks
        """
        chunks = []
        
        # If sequence is too short, skip
        if len(tokens) < self.min_length:
            return chunks
        
        # If sequence fits in max_length, use as is
        if len(tokens) <= self.max_length:
            chunks.append(tokens)
            return chunks
        
        # Otherwise, chunk into overlapping sequences
        start = 0
        while start < len(tokens):
            end = min(start + self.target_length, len(tokens))
            chunk = tokens[start:end]
            
            if len(chunk) >= self.min_length:
                chunks.append(chunk)
            
            # Overlap by 50% for better coverage
            start += self.target_length // 2
            
            if end >= len(tokens):
                break
        
        return chunks
    
    def process_dataset(
        self,
        dataset_config: DatasetConfig,
        max_samples: Optional[int] = None
    ) -> Iterator[List[int]]:
        """
        Process a single dataset
        
        Args:
            dataset_config: Dataset configuration
            max_samples: Maximum number of samples to process (None for all)
            
        Yields:
            Lists of token IDs
        """
        # Load dataset
        try:
            if dataset_config.streaming:
                if dataset_config.config:
                    dataset = load_dataset(
                        dataset_config.path,
                        dataset_config.config,
                        streaming=True,
                        split="train"
                    )
                elif dataset_config.subset:
                    dataset = load_dataset(
                        dataset_config.path,
                        dataset_config.subset,
                        streaming=True,
                        split="train"
                    )
                else:
                    dataset = load_dataset(
                        dataset_config.path,
                        streaming=True,
                        split="train"
                    )
            else:
                if dataset_config.config:
                    dataset = load_dataset(
                        dataset_config.path,
                        dataset_config.config,
                        split="train"
                    )
                else:
                    dataset = load_dataset(
                        dataset_config.path,
                        split="train"
                    )
        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_config.name}: {e}")
            return
        
        # Process dataset
        sample_count = 0
        for example in dataset:
            if max_samples and sample_count >= max_samples:
                break
            
            text = example.get(dataset_config.text_column, '')
            if not text:
                continue
            
            # Filter German text
            if not self.filter_german(example):
                continue
            
            # Tokenize
            tokens = self.tokenize_text(text)
            
            # Chunk tokens
            chunks = self.chunk_tokens(tokens)
            
            # Yield chunks
            for chunk in chunks:
                yield chunk
                sample_count += 1
                
                if max_samples and sample_count >= max_samples:
                    break
    
    def process_multiple_datasets(
        self,
        dataset_configs: List[DatasetConfig],
        target_tokens: Optional[int] = None,
        weights: Optional[List[float]] = None
    ) -> Iterator[List[int]]:
        """
        Process multiple datasets with weighted mixing
        
        Args:
            dataset_configs: List of dataset configurations
            target_tokens: Target number of tokens to generate
            weights: Optional weights for each dataset (if None, uses dataset_config.weight)
            
        Yields:
            Lists of token IDs
        """
        if weights is None:
            weights = [cfg.weight for cfg in dataset_configs]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Create iterators for each dataset
        iterators = []
        for cfg in dataset_configs:
            # Calculate max_samples based on weight
            max_samples = None
            if target_tokens:
                # Rough estimate: assume average chunk length
                avg_chunk_length = self.target_length
                target_samples = int(target_tokens / avg_chunk_length * cfg.weight)
                max_samples = target_samples
            
            iterator = self.process_dataset(cfg, max_samples=max_samples)
            iterators.append(iterator)
        
        # Mix datasets according to weights
        # Simple round-robin with weights
        token_count = 0
        dataset_indices = list(range(len(dataset_configs)))
        
        # Create weighted selection
        import random
        while True:
            if target_tokens and token_count >= target_tokens:
                break
            
            # Select dataset based on weights
            selected_idx = random.choices(dataset_indices, weights=weights)[0]
            
            try:
                chunk = next(iterators[selected_idx])
                token_count += len(chunk)
                yield chunk
            except StopIteration:
                # Remove exhausted iterator
                iterators[selected_idx] = None
                if all(it is None for it in iterators):
                    break
        
        print(f"Processed {token_count:,} tokens from {len(dataset_configs)} datasets")


def save_tokenized_dataset(tokens: List[List[int]], output_path: str):
    """
    Save tokenized dataset to disk
    
    Args:
        tokens: List of token sequences
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten tokens
    flat_tokens = []
    for seq in tokens:
        flat_tokens.extend(seq)
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flat_tokens, f)
    
    print(f"Saved {len(flat_tokens):,} tokens to {output_path}")


def load_tokenized_dataset(input_path: str) -> List[int]:
    """
    Load tokenized dataset from disk
    
    Args:
        input_path: Input file path
        
    Returns:
        List of token IDs
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    
    return tokens

