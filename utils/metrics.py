"""
Metrics tracking and visualization utilities
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self, metrics_file: str):
        """
        Initialize metrics tracker
        
        Args:
            metrics_file: Path to JSON file for storing metrics
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics: List[Dict] = []
        
        # Load existing metrics if file exists
        if self.metrics_file.exists():
            self.load()
    
    def log(self, metrics: Dict):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics to log
        """
        self.metrics.append(metrics)
        self.save()
    
    def save(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self):
        """Load metrics from file"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)
    
    def get_latest(self) -> Optional[Dict]:
        """Get latest metrics"""
        return self.metrics[-1] if self.metrics else None
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric"""
        return [m.get(metric_name) for m in self.metrics if metric_name in m]


def save_metrics(metrics: Dict, metrics_file: str):
    """
    Save metrics to file
    
    Args:
        metrics: Metrics dictionary
        metrics_file: Path to metrics file
    """
    metrics_file = Path(metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics
    existing_metrics = []
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            existing_metrics = json.load(f)
    
    # Append new metrics
    existing_metrics.append(metrics)
    
    # Save
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(existing_metrics, f, indent=2)


def load_metrics(metrics_file: str) -> List[Dict]:
    """
    Load metrics from file
    
    Args:
        metrics_file: Path to metrics file
        
    Returns:
        List of metrics dictionaries
    """
    metrics_file = Path(metrics_file)
    if not metrics_file.exists():
        return []
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_curves(
    metrics_file: str,
    output_path: str,
    metrics_to_plot: Optional[List[str]] = None
):
    """
    Plot training curves from metrics file
    
    Args:
        metrics_file: Path to metrics JSON file
        output_path: Path to save plot
        metrics_to_plot: List of metric names to plot (default: ['loss', 'perplexity'])
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['loss', 'perplexity']
    
    metrics = load_metrics(metrics_file)
    if not metrics:
        print("No metrics to plot")
        return
    
    # Check if steps exist
    if not any('step' in m for m in metrics):
        print("No step data found in metrics")
        return
    
    # Extract steps and metrics
    steps = [m.get('step', i) for i, m in enumerate(metrics)]
    
    # Create subplots
    num_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    for idx, metric_name in enumerate(metrics_to_plot):
        values = [m.get(metric_name) for m in metrics if metric_name in m]
        metric_steps = [s for s, m in zip(steps, metrics) if metric_name in m]
        
        if values:
            axes[idx].plot(metric_steps, values, linewidth=2)
            axes[idx].set_xlabel('Step')
            axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
            axes[idx].set_title(f'{metric_name.replace("_", " ").title()} over Training')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_path}")


def plot_comprehensive_metrics(
    metrics_file: str,
    output_path: str
):
    """
    Plot comprehensive training metrics including loss, perplexity, LR, VRAM
    
    Args:
        metrics_file: Path to metrics JSON file
        output_path: Path to save plot
    """
    metrics = load_metrics(metrics_file)
    if not metrics:
        print("No metrics to plot")
        return
    
    # Check if steps exist
    if not any('step' in m for m in metrics):
        print("No step data found in metrics")
        return
    
    # Extract data
    steps = [m.get('step', i) for i, m in enumerate(metrics)]
    losses = [m.get('loss') for m in metrics if 'loss' in m]
    perplexities = [m.get('perplexity') for m in metrics if 'perplexity' in m]
    lrs = [m.get('learning_rate') for m in metrics if 'learning_rate' in m]
    vram = [m.get('vram_usage') for m in metrics if 'vram_usage' in m]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    if losses:
        loss_steps = [s for s, m in zip(steps, metrics) if 'loss' in m]
        axes[0, 0].plot(loss_steps, losses, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Perplexity curve
    if perplexities:
        ppl_steps = [s for s, m in zip(steps, metrics) if 'perplexity' in m]
        axes[0, 1].plot(ppl_steps, perplexities, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate curve
    if lrs:
        lr_steps = [s for s, m in zip(steps, metrics) if 'learning_rate' in m]
        axes[1, 0].plot(lr_steps, lrs, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # VRAM usage
    if vram:
        vram_steps = [s for s, m in zip(steps, metrics) if 'vram_usage' in m]
        axes[1, 1].plot(vram_steps, vram, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('VRAM Usage (GB)')
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive metrics plot to {output_path}")

