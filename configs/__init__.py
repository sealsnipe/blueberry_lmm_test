"""
Configuration loader for German training
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration settings
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Validate model config
    assert 'model' in config, "Missing 'model' section in config"
    assert config['model']['d_model'] > 0, "d_model must be positive"
    assert config['model']['n_layers'] > 0, "n_layers must be positive"
    
    # Validate training config
    assert 'training' in config, "Missing 'training' section in config"
    assert config['training']['batch_size'] > 0, "batch_size must be positive"
    assert config['training']['learning_rate'] > 0, "learning_rate must be positive"
    
    # Validate dataset config
    assert 'dataset' in config, "Missing 'dataset' section in config"
    assert len(config['dataset']['datasets']) > 0, "Must specify at least one dataset"
    
    return True

