#!/usr/bin/env python3
"""
Quick Start Script for Production Training
Optimized for RTX 3090 - Ready to go!
"""
import subprocess
import sys
import os
from pathlib import Path

def check_setup():
    """Verify all prerequisites"""
    print("=" * 60)
    print("Blueberry DE-Train - Production Setup Check")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - training will be slow on CPU")
    except ImportError:
        print("⚠️  PyTorch not installed - run: pip install -r requirements.txt")
        return False
    
    # Check config file
    config_path = Path("configs/de_training.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    print(f"✅ Config file found: {config_path}")
    
    # Check dependencies
    required = ['torch', 'transformers', 'datasets', 'accelerate']
    missing = []
    for dep in required:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    
    # Check Wandb (optional)
    try:
        import wandb
        if wandb.api.api_key is None:
            print("⚠️  Wandb not logged in - using offline mode")
            print("   For online logging: wandb login")
        else:
            print("✅ Wandb configured")
    except ImportError:
        print("⚠️  Wandb not installed - using local logging only")
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete - Ready for Production!")
    print("=" * 60)
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quick start production training on RTX 3090"
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run preprocessing (test dataset)"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100000,
        help="Preprocessing subset size for testing (default: 100000)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Wandb mode (default: online)"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip setup verification"
    )
    
    args = parser.parse_args()
    
    # Setup check
    if not args.skip_check:
        if not check_setup():
            sys.exit(1)
    
    # Preprocessing step
    if args.preprocess_only:
        print("\n" + "=" * 60)
        print("Running Preprocessing (Test Subset)")
        print("=" * 60)
        cmd = [
            sys.executable,
            "-m", "data.preprocess",
            "--config", "configs/de_training.yaml",
            "--subset_size", str(args.subset_size)
        ]
        subprocess.run(cmd)
        print("\n✅ Preprocessing complete!")
        return
    
    # Full training
    print("\n" + "=" * 60)
    print("Starting Production Training")
    print("=" * 60)
    print(f"Device: CUDA:{args.device}")
    print(f"Wandb Mode: {args.wandb_mode}")
    print("\nTip: Open another terminal and run:")
    print("  python monitoring/live_metrics.py --log_path ./logs/metrics.json")
    print("\n" + "=" * 60)
    
    cmd = [
        sys.executable,
        "run_full.py",
        "--config", "configs/de_training.yaml",
        "--device", str(args.device),
        "--wandb_mode", args.wandb_mode
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

