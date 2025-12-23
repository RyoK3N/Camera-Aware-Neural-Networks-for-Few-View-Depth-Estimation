#!/usr/bin/env python
"""
TensorBoard Launcher Script
Automatically launches TensorBoard with proper configuration
Uses conda environment's Python
"""

import argparse
import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path


def check_tensorboard_installed():
    """Check if TensorBoard is installed"""
    try:
        result = subprocess.run(
            ['tensorboard', '--version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def launch_tensorboard(logdir, port=6006, open_browser=True, bind_all=False):
    """Launch TensorBoard with specified configuration"""

    # Verify logdir exists
    logdir_path = Path(logdir)
    if not logdir_path.exists():
        print(f"⚠️  Warning: Log directory '{logdir}' does not exist yet.")
        print(f"   Creating directory...")
        logdir_path.mkdir(parents=True, exist_ok=True)

    # Build TensorBoard command
    cmd = [
        'tensorboard',
        '--logdir', str(logdir),
        '--port', str(port),
        '--reload_interval', '5',  # Refresh every 5 seconds
    ]

    if bind_all:
        cmd.append('--bind_all')

    print("=" * 70)
    print("🚀 Launching TensorBoard")
    print("=" * 70)
    print(f"📁 Log Directory: {logdir_path.absolute()}")
    print(f"🌐 Port: {port}")
    print(f"🔗 URL: http://localhost:{port}")
    if bind_all:
        print(f"🌍 Network Access: Enabled (accessible from other machines)")
    print("=" * 70)
    print()
    print("💡 Tip: Keep this terminal open while training")
    print("   Press Ctrl+C to stop TensorBoard")
    print()

    # Open browser after delay
    if open_browser:
        print("🌐 Opening TensorBoard in browser in 3 seconds...")

        def open_browser_delayed():
            time.sleep(3)
            url = f"http://localhost:{port}"
            try:
                webbrowser.open(url)
                print(f"✅ Browser opened: {url}")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
                print(f"   Please open manually: {url}")

        # Open browser in background
        import threading
        browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
        browser_thread.start()

    # Launch TensorBoard
    try:
        print("▶️  Starting TensorBoard...")
        print()
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n")
        print("⏹️  TensorBoard stopped")
        print()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching TensorBoard: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Launch TensorBoard for monitoring training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/launch_tensorboard.py

  # Custom port
  python scripts/launch_tensorboard.py --port 8080

  # Different log directory
  python scripts/launch_tensorboard.py --logdir ./runs/experiment1

  # Allow network access (for remote monitoring)
  python scripts/launch_tensorboard.py --bind-all

  # Don't open browser automatically
  python scripts/launch_tensorboard.py --no-browser
        """
    )

    parser.add_argument(
        '--logdir',
        type=str,
        default='./runs',
        help='TensorBoard log directory (default: ./runs)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='Port to run TensorBoard on (default: 6006)'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )

    parser.add_argument(
        '--bind-all',
        action='store_true',
        help='Allow access from other machines (bind to 0.0.0.0)'
    )

    args = parser.parse_args()

    # Check if TensorBoard is installed
    if not check_tensorboard_installed():
        print("❌ Error: TensorBoard is not installed")
        print()
        print("Install with:")
        print("  pip install tensorboard")
        print("  # or")
        print("  conda install tensorboard")
        sys.exit(1)

    # Launch TensorBoard
    launch_tensorboard(
        logdir=args.logdir,
        port=args.port,
        open_browser=not args.no_browser,
        bind_all=args.bind_all
    )


if __name__ == '__main__':
    main()
