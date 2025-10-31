"""
Full training pipeline wrapper
Starts training and monitoring in parallel
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from configs import load_config


def run_training(config_path: str, device_id: int = 0, wandb_mode: str = "online", resume_from: str = None):
    """Start training process"""
    cmd = [
        sys.executable,
        "training/de_train.py",
        "--config", config_path,
        "--device", str(device_id),
        "--wandb_mode", wandb_mode
    ]
    
    if resume_from:
        cmd.extend(["--resume_from", resume_from])
    
    print(f"Starting training: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    return process


def run_monitoring(metrics_file: str, interval: float = 30.0, use_watchdog: bool = False):
    """Start monitoring process"""
    cmd = [
        sys.executable,
        "monitoring/live_metrics.py",
        "--log_path", metrics_file,
        "--interval", str(interval)
    ]
    
    if use_watchdog:
        cmd.append("--watchdog")
    
    print(f"Starting monitoring: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    
    return process


def run_full(config_path: str, device_id: int = 0, wandb_mode: str = "online", 
             resume_from: str = None, monitor: bool = True, monitoring_interval: float = 30.0):
    """
    Run full training pipeline with monitoring
    
    Args:
        config_path: Path to config YAML file
        device_id: CUDA device ID
        wandb_mode: Wandb mode
        resume_from: Path to checkpoint to resume from
        monitor: Whether to start monitoring
        monitoring_interval: Monitoring polling interval
    """
    # Load config to get metrics file path
    config = load_config(config_path)
    metrics_file = config['logging']['metrics_file']
    
    print("=" * 60)
    print("German PLASA LLM Training Pipeline")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Device: CUDA:{device_id}")
    print(f"Wandb Mode: {wandb_mode}")
    print("=" * 60)
    
    # Start training
    training_process = run_training(config_path, device_id, wandb_mode, resume_from)
    
    # Start monitoring
    monitoring_process = None
    if monitor:
        # Wait a bit for metrics file to be created
        time.sleep(5)
        monitoring_process = run_monitoring(metrics_file, monitoring_interval)
    
    try:
        # Wait for training to complete
        print("\nTraining started. Press Ctrl+C to stop both processes.\n")
        
        # Stream training output
        for line in training_process.stdout:
            print(line, end='')
        
        training_process.wait()
        
        # Stop monitoring
        if monitoring_process:
            monitoring_process.terminate()
            monitoring_process.wait()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nStopping training and monitoring...")
        training_process.terminate()
        if monitoring_process:
            monitoring_process.terminate()
        
        training_process.wait()
        if monitoring_process:
            monitoring_process.wait()
        
        print("Processes stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full training pipeline with monitoring")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--wandb_mode", type=str, default="online",
                       choices=["online", "offline", "disabled"],
                       help="Wandb mode")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--no_monitor", action="store_true",
                       help="Disable monitoring")
    parser.add_argument("--monitoring_interval", type=float, default=30.0,
                       help="Monitoring polling interval in seconds")
    
    args = parser.parse_args()
    
    run_full(
        config_path=args.config,
        device_id=args.device,
        wandb_mode=args.wandb_mode,
        resume_from=args.resume_from,
        monitor=not args.no_monitor,
        monitoring_interval=args.monitoring_interval
    )

