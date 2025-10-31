"""
Integration tests
"""
import pytest
import torch
import tempfile
from pathlib import Path

from configs import load_config, validate_config
from models import create_plasa_model
from utils.metrics import MetricsTracker


def test_config_loading():
    """Test config loading and validation"""
    config_path = "configs/de_training.yaml"
    
    # Check file exists
    assert Path(config_path).exists()
    
    config = load_config(config_path)
    assert config is not None
    assert 'model' in config
    assert 'training' in config
    assert 'dataset' in config
    
    # Validate config
    validate_config(config)


def test_training_loop_mini():
    """Test mini training loop"""
    # Create small model
    config = {
        'model': {
            'vocab_size': 1000,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 256,
            'max_seq_len': 128,
            'attention': {
                'indexer_heads': 2,
                'indexer_dim': 16,
                'sparse_top_k': 64
            },
            'dropout': 0.1
        }
    }
    
    model = create_plasa_model(config)
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create dummy data
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, 1000, (batch_size, seq_len))
    y = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    logits, _ = model(x)
    
    # Loss calculation
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )
    
    assert loss.item() > 0
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad


def test_checkpoint_save_load():
    """Test checkpoint saving and loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Create small model
        config = {
            'model': {
                'vocab_size': 1000,
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 2,
                'd_ff': 256,
                'max_seq_len': 128,
                'attention': {
                    'indexer_heads': 2,
                    'indexer_dim': 16,
                    'sparse_top_k': 64
                },
                'dropout': 0.1
            }
        }
        
        model = create_plasa_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)
        
        # Save checkpoint
        checkpoint = {
            'step': 100,
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': {'loss': 2.5}
        }
        
        checkpoint_path = checkpoint_dir / "test.pt"
        torch.save(checkpoint, checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert loaded_checkpoint['step'] == 100
        assert loaded_checkpoint['epoch'] == 1
        assert 'model_state_dict' in loaded_checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

