#!/usr/bin/env python3
"""
Quick start script for German PLASA LLM training
"""
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['torch', 'transformers', 'datasets', 'accelerate']
    missing = []
    
    for dep in required:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    print("=" * 60)
    print("German PLASA LLM Training - Quick Start")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check config file
    config_path = Path("configs/de_training.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available. Training will run on CPU (slow!)")
    except Exception as e:
        print(f"Warning: Could not check CUDA: {e}")
    
    # Optional wandb login (skip if offline)
    try:
        import wandb
        if wandb.api.api_key is None:
            print("\n⚠ Wandb not logged in. Use --wandb_mode offline or run 'wandb login'")
    except Exception:
        pass  # Wandb not available or offline mode
    
    # Run training
    print("\nStarting training pipeline...")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        "run_full.py",
        "--config", str(config_path),
        "--device", "0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Training failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

