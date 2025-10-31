#!/usr/bin/env python3
"""
Helper script to resume training from checkpoint
"""
import argparse
import sys
from pathlib import Path

def find_latest_checkpoint(checkpoint_dir: str = "./checkpoints"):
    """Find the latest checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Look for latest.pt first
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return str(latest)
    
    # Otherwise find checkpoint with highest step number
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    
    # Extract step numbers and find max
    def get_step(path):
        try:
            return int(path.stem.split("_")[-1])
        except:
            return 0
    
    latest_checkpoint = max(checkpoints, key=get_step)
    return str(latest_checkpoint)

def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                       help="Directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint to resume from (default: latest)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--wandb_mode", type=str, default="online",
                       choices=["online", "offline", "disabled"],
                       help="Wandb mode")
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            print(f"Found latest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found. Starting fresh training.")
            checkpoint_path = None
    
    # Run training with resume
    cmd = [
        sys.executable,
        "training/de_train.py",
        "--config", args.config,
        "--device", str(args.device),
        "--wandb_mode", args.wandb_mode
    ]
    
    if checkpoint_path:
        cmd.extend(["--resume_from", checkpoint_path])
    
    print(f"Resuming training with command: {' '.join(cmd)}")
    print("=" * 60)
    
    import subprocess
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

