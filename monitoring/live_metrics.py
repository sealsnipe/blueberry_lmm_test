"""
Asynchronous monitoring script for live training metrics
Polls metrics.json and displays live updates
"""
import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not available. Using polling mode.")


class MetricsFileHandler(FileSystemEventHandler):
    """Watchdog handler for metrics file changes"""
    def __init__(self, callback):
        self.callback = callback
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('metrics.json'):
            self.callback()


class MetricsMonitor:
    """Monitor training metrics and display live updates"""
    
    def __init__(self, metrics_file: str, interval: float = 30.0):
        """
        Initialize monitor
        
        Args:
            metrics_file: Path to metrics JSON file
            interval: Polling interval in seconds (if watchdog not available)
        """
        self.metrics_file = Path(metrics_file)
        self.interval = interval
        self.last_metrics = None
        self.last_step = 0
        
    def load_metrics(self) -> Optional[Dict]:
        """Load latest metrics from file"""
        if not self.metrics_file.exists():
            return None
        
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                metrics_list = json.load(f)
            
            if not metrics_list:
                return None
            
            # Return latest metrics (last dict in list)
            return metrics_list[-1]
        except (json.JSONDecodeError, IOError, IndexError) as e:
            print(f"Error reading metrics: {e}")
            return None
    
    def get_trend_arrow(self, current: float, previous: float) -> str:
        """
        Get trend arrow for metric
        
        Args:
            current: Current value
            previous: Previous value
            
        Returns:
            Arrow emoji
        """
        if current < previous:
            return "↓"
        elif current > previous:
            return "↑"
        else:
            return "→"
    
    def format_metrics(self, metrics: Dict, previous: Optional[Dict] = None) -> str:
        """
        Format metrics for display
        
        Args:
            metrics: Current metrics
            previous: Previous metrics for trend calculation
            
        Returns:
            Formatted string
        """
        step = metrics.get('step', 0)
        epoch = metrics.get('epoch', 0)
        loss = metrics.get('loss', 0.0)
        perplexity = metrics.get('perplexity', 0.0)
        lr = metrics.get('learning_rate', 0.0)
        vram = metrics.get('vram_usage', 0.0)
        tokens_per_sec = metrics.get('tokens_per_second', 0.0)
        
        # Calculate trends (only if previous metrics exist)
        loss_arrow = ""
        ppl_arrow = ""
        tokens_arrow = ""
        if previous:
            if 'loss' in previous:
                loss_arrow = self.get_trend_arrow(loss, previous.get('loss', loss))
            if 'perplexity' in previous:
                ppl_arrow = self.get_trend_arrow(perplexity, previous.get('perplexity', perplexity))
            if 'tokens_per_second' in previous:
                tokens_arrow = self.get_trend_arrow(tokens_per_sec, previous.get('tokens_per_second', tokens_per_sec))
        
        # Format output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        output = f"""
╔════════════════════════════════════════════════════════════════╗
║  Training Metrics Monitor - {timestamp:<20}     ║
╠════════════════════════════════════════════════════════════════╣
║  Step: {step:<10} Epoch: {epoch:<6}                               ║
║  Loss: {loss:.4f} {loss_arrow:<2}  Perplexity: {perplexity:.2f} {ppl_arrow:<2}            ║
║  Learning Rate: {lr:.2e}                                     ║
║  VRAM Usage: {vram:.2f} GB                                      ║
║  Tokens/sec: {tokens_per_sec:.0f} {tokens_arrow:<2}                                 ║
╚════════════════════════════════════════════════════════════════╝
"""
        return output
    
    def display_metrics(self, metrics: Dict):
        """Display metrics in terminal"""
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Format and print
        previous = self.last_metrics if self.last_step < metrics.get('step', 0) else None
        formatted = self.format_metrics(metrics, previous)
        print(formatted)
        
        # Update last metrics
        self.last_metrics = metrics.copy()
        self.last_step = metrics.get('step', 0)
    
    def run_polling(self):
        """Run monitoring with polling"""
        print(f"Starting metrics monitor (polling every {self.interval}s)...")
        print(f"Monitoring: {self.metrics_file}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                metrics = self.load_metrics()
                
                if metrics:
                    # Only update if step changed
                    current_step = metrics.get('step', 0)
                    if current_step > self.last_step:
                        self.display_metrics(metrics)
                
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    def run_watchdog(self):
        """Run monitoring with watchdog (file system events)"""
        if not WATCHDOG_AVAILABLE:
            print("Watchdog not available, falling back to polling mode")
            self.run_polling()
            return
        
        print(f"Starting metrics monitor (file watching)...")
        print(f"Monitoring: {self.metrics_file}")
        print("Press Ctrl+C to stop\n")
        
        # Setup watchdog
        event_handler = MetricsFileHandler(self.on_metrics_updated)
        observer = Observer()
        observer.schedule(event_handler, path=str(self.metrics_file.parent), recursive=False)
        observer.start()
        
        # Initial display
        metrics = self.load_metrics()
        if metrics:
            self.display_metrics(metrics)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n\nMonitoring stopped.")
        
        observer.join()
    
    def on_metrics_updated(self):
        """Callback when metrics file is updated"""
        metrics = self.load_metrics()
        if metrics:
            current_step = metrics.get('step', 0)
            if current_step > self.last_step:
                self.display_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="Live metrics monitor for training")
    parser.add_argument("--log_path", type=str, default="./logs/metrics.json",
                       help="Path to metrics JSON file")
    parser.add_argument("--interval", type=float, default=30.0,
                       help="Polling interval in seconds")
    parser.add_argument("--watchdog", action="store_true",
                       help="Use file system watching instead of polling")
    
    args = parser.parse_args()
    
    monitor = MetricsMonitor(args.log_path, args.interval)
    
    if args.watchdog and WATCHDOG_AVAILABLE:
        monitor.run_watchdog()
    else:
        monitor.run_polling()


if __name__ == "__main__":
    main()

