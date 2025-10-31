"""
Unit tests for preprocessing and training components
"""
import pytest
import torch
import tempfile
import json
from pathlib import Path

from data import GermanDatasetPreprocessor, DatasetConfig, save_tokenized_dataset, load_tokenized_dataset
from models import create_plasa_model
from utils.metrics import MetricsTracker, plot_training_curves


def test_dataset_config():
    """Test DatasetConfig dataclass"""
    config = DatasetConfig(
        name="test",
        path="test/path",
        weight=0.5,
        streaming=True
    )
    
    assert config.name == "test"
    assert config.path == "test/path"
    assert config.weight == 0.5
    assert config.streaming is True


def test_preprocessor_init():
    """Test preprocessor initialization"""
    preprocessor = GermanDatasetPreprocessor(
        tokenizer_name="gpt2",
        min_length=256,
        max_length=1024,
        target_length=512
    )
    
    assert preprocessor.min_length == 256
    assert preprocessor.max_length == 1024
    assert preprocessor.target_length == 512


def test_tokenize_text():
    """Test text tokenization"""
    preprocessor = GermanDatasetPreprocessor()
    
    text = "Dies ist ein deutscher Text."
    tokens = preprocessor.tokenize_text(text)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_chunk_tokens():
    """Test token chunking"""
    preprocessor = GermanDatasetPreprocessor(
        min_length=10,
        max_length=100,
        target_length=50
    )
    
    # Short sequence
    short_tokens = [1] * 5
    chunks = preprocessor.chunk_tokens(short_tokens)
    assert len(chunks) == 0
    
    # Long sequence
    long_tokens = list(range(200))
    chunks = preprocessor.chunk_tokens(long_tokens)
    assert len(chunks) > 0
    assert all(len(chunk) >= preprocessor.min_length for chunk in chunks)
    assert all(len(chunk) <= preprocessor.max_length for chunk in chunks)


def test_save_load_tokenized_dataset():
    """Test saving and loading tokenized dataset"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_tokens.json"
        
        tokens = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        # Flatten for save function
        flat_tokens = []
        for seq in tokens:
            flat_tokens.extend(seq)
        
        save_tokenized_dataset(flat_tokens, str(output_path))
        assert output_path.exists()
        
        loaded_tokens = load_tokenized_dataset(str(output_path))
        assert loaded_tokens == flat_tokens


def test_plasa_model_creation():
    """Test PLASA model creation"""
    config = {
        'model': {
            'vocab_size': 1000,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 512,
            'max_seq_len': 256,
            'attention': {
                'indexer_heads': 2,
                'indexer_dim': 32,
                'sparse_top_k': 128
            },
            'dropout': 0.1
        }
    }
    
    model = create_plasa_model(config)
    
    assert model is not None
    assert model.vocab_size == 1000
    assert model.d_model == 128
    assert model.n_layers == 2
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits, _ = model(x)
    
    assert logits.shape == (batch_size, seq_len, 1000)


def test_metrics_tracker():
    """Test metrics tracker"""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_file = Path(tmpdir) / "metrics.json"
        
        tracker = MetricsTracker(str(metrics_file))
        
        # Log metrics
        tracker.log({'step': 1, 'loss': 2.5})
        tracker.log({'step': 2, 'loss': 2.3})
        
        # Check file exists
        assert metrics_file.exists()
        
        # Load and verify
        tracker2 = MetricsTracker(str(metrics_file))
        assert len(tracker2.metrics) == 2
        assert tracker2.get_latest()['step'] == 2
        
        # Test metric history
        loss_history = tracker2.get_metric_history('loss')
        assert loss_history == [2.5, 2.3]


def test_plot_training_curves():
    """Test plotting training curves"""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_file = Path(tmpdir) / "metrics.json"
        output_path = Path(tmpdir) / "plot.png"
        
        # Create dummy metrics
        metrics = [
            {'step': 1, 'loss': 2.5, 'perplexity': 12.0},
            {'step': 2, 'loss': 2.3, 'perplexity': 10.0},
            {'step': 3, 'loss': 2.1, 'perplexity': 8.0}
        ]
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        # Plot (should not raise exception)
        try:
            plot_training_curves(str(metrics_file), str(output_path))
            # File might not be created if matplotlib backend is not available
            # Just check no exception was raised
            assert True
        except Exception as e:
            # If plotting fails due to display issues, that's okay for CI
            if "display" in str(e).lower() or "backend" in str(e).lower():
                pass
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

