#!/usr/bin/env python
"""
Real-time Training Monitor
Displays live training logs with formatted output and progress tracking
Uses conda environment's Python
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime
import re


class TrainingMonitor:
    def __init__(self, log_file, metrics_file=None):
        self.log_file = Path(log_file)
        self.metrics_file = Path(metrics_file) if metrics_file else None
        self.last_position = 0

    def clear_screen(self):
        """Clear terminal screen"""
        print("\033[2J\033[H", end='')

    def print_header(self):
        """Print monitoring header"""
        print("=" * 80)
        print("📊 TRAINING MONITOR - Real-time Logs".center(80))
        print("=" * 80)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Log file: {self.log_file}")
        print("-" * 80)
        print()

    def parse_log_line(self, line):
        """Parse and format log line"""
        # Color codes
        BOLD = '\033[1m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        RESET = '\033[0m'

        # Highlight epochs
        if 'Epoch' in line and '---' in line:
            return f"{BOLD}{BLUE}{line}{RESET}"

        # Highlight training progress
        if 'Training:' in line or '[' in line and '%]' in line:
            return f"{GREEN}{line}{RESET}"

        # Highlight validation
        if 'Validation:' in line or 'Val Loss' in line:
            return f"{CYAN}{line}{RESET}"

        # Highlight warnings
        if 'Warning' in line or 'warning' in line:
            return f"{YELLOW}{line}{RESET}"

        # Highlight errors
        if 'Error' in line or 'error' in line or 'failed' in line:
            return f"{RED}{line}{RESET}"

        # Highlight metrics
        if any(metric in line for metric in ['abs_rel', 'rmse', 'Loss:', 'δ<1.25']):
            return f"{BOLD}{line}{RESET}"

        return line

    def tail_log(self, follow=True, lines=50):
        """Tail log file with optional follow mode"""

        if not self.log_file.exists():
            print(f"⏳ Waiting for log file: {self.log_file}")
            while not self.log_file.exists():
                time.sleep(1)
            print(f"✅ Log file found!")
            print()

        # Print initial lines
        with open(self.log_file, 'r') as f:
            # Get last N lines
            all_lines = f.readlines()
            start_idx = max(0, len(all_lines) - lines)
            for line in all_lines[start_idx:]:
                print(self.parse_log_line(line.rstrip()))

            self.last_position = f.tell()

        if not follow:
            return

        print()
        print("👀 Following log file... (Press Ctrl+C to stop)")
        print()

        # Follow mode
        try:
            while True:
                with open(self.log_file, 'r') as f:
                    f.seek(self.last_position)
                    new_lines = f.readlines()

                    for line in new_lines:
                        print(self.parse_log_line(line.rstrip()))

                    self.last_position = f.tell()

                time.sleep(0.5)  # Check every 0.5 seconds
        except KeyboardInterrupt:
            print()
            print("⏹️  Monitoring stopped")

    def show_metrics_summary(self):
        """Show summary of metrics from CSV"""
        if not self.metrics_file or not self.metrics_file.exists():
            return

        print()
        print("=" * 80)
        print("📈 LATEST METRICS SUMMARY".center(80))
        print("=" * 80)

        try:
            with open(self.metrics_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Parse header
                    header = lines[0].strip().split(',')
                    # Get last row
                    last_row = lines[-1].strip().split(',')

                    print()
                    for col, val in zip(header, last_row):
                        if col in ['epoch', 'step']:
                            print(f"  {col:15s}: {val}")
                        elif 'loss' in col.lower():
                            print(f"  {col:15s}: {float(val):.6f}")
                        else:
                            try:
                                print(f"  {col:15s}: {float(val):.4f}")
                            except:
                                print(f"  {col:15s}: {val}")
                    print()
        except Exception as e:
            print(f"⚠️  Could not parse metrics: {e}")

    def live_dashboard(self, refresh_interval=5):
        """Live dashboard with metrics and recent logs"""
        try:
            while True:
                self.clear_screen()
                self.print_header()

                # Show recent logs
                print("📝 RECENT LOGS (last 20 lines):")
                print("-" * 80)

                if self.log_file.exists():
                    with open(self.log_file, 'r') as f:
                        lines = f.readlines()
                        start_idx = max(0, len(lines) - 20)
                        for line in lines[start_idx:]:
                            print(self.parse_log_line(line.rstrip()))
                else:
                    print("⏳ Waiting for log file...")

                # Show metrics summary
                self.show_metrics_summary()

                print()
                print(f"🔄 Refreshing every {refresh_interval}s... (Press Ctrl+C to stop)")

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print()
            print("⏹️  Dashboard stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor training logs in real-time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Follow training logs (tail -f style)
  python scripts/monitor_training.py

  # Live dashboard (auto-refreshing)
  python scripts/monitor_training.py --dashboard

  # Show last 100 lines
  python scripts/monitor_training.py --lines 100

  # Custom log file
  python scripts/monitor_training.py --log logs/custom_training.log

  # With metrics
  python scripts/monitor_training.py --metrics logs/metrics.csv
        """
    )

    parser.add_argument(
        '--log',
        type=str,
        default='./logs/training.log',
        help='Training log file (default: ./logs/training.log)'
    )

    parser.add_argument(
        '--metrics',
        type=str,
        default='./logs/metrics.csv',
        help='Metrics CSV file (default: ./logs/metrics.csv)'
    )

    parser.add_argument(
        '--lines',
        type=int,
        default=50,
        help='Number of initial lines to show (default: 50)'
    )

    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Show live dashboard with auto-refresh'
    )

    parser.add_argument(
        '--refresh',
        type=int,
        default=5,
        help='Dashboard refresh interval in seconds (default: 5)'
    )

    parser.add_argument(
        '--no-follow',
        action='store_true',
        help='Do not follow log file (just print and exit)'
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(args.log, args.metrics)

    if args.dashboard:
        monitor.live_dashboard(refresh_interval=args.refresh)
    else:
        monitor.tail_log(follow=not args.no_follow, lines=args.lines)


if __name__ == '__main__':
    main()
