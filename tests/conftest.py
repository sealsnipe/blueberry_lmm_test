"""
Pytest configuration
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def temp_dir():
    """Fixture for temporary directory"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_config():
    """Fixture for sample config"""
    return {
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

