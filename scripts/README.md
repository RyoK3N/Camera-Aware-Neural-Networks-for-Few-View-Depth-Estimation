# Training Scripts and Monitoring Tools

Complete set of scripts for training, monitoring, and visualizing depth estimation models with automatic TensorBoard integration.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Script Overview](#script-overview)
- [Training Scripts](#training-scripts)
- [Monitoring Scripts](#monitoring-scripts)
- [TensorBoard Scripts](#tensorboard-scripts)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Mac M4 Pro (Easiest Way)

```bash
# One command to start everything:
# - Training with TensorBoard
# - Auto-opens browser
# - Live log monitoring
./scripts/quick_train_m4pro.sh
```

### Production GPU

```bash
# One command for production training
./scripts/quick_train_production.sh
```

### Manual Control

```bash
# 1. Start TensorBoard (separate terminal)
./scripts/start_tensorboard.sh 6006

# 2. Monitor logs (separate terminal)
./scripts/watch_training.sh

# 3. Run training (main terminal)
./build/train --config configs/train_config_m4pro.yaml --tensorboard true
```

## 📦 Script Overview

| Script | Purpose | Auto-launches |
|--------|---------|---------------|
| `quick_train_m4pro.sh` | Mac M4 Pro one-click training | TensorBoard + Monitor |
| `quick_train_production.sh` | Production GPU one-click training | TensorBoard + Monitor |
| `train_with_monitoring.sh` | Full orchestration (any config) | TensorBoard + Monitor |
| `start_tensorboard.sh` | TensorBoard only | Browser |
| `watch_training.sh` | Log monitoring only | - |
| `launch_tensorboard.py` | Python TensorBoard launcher | Browser |
| `monitor_training.py` | Python log monitor | - |

## 🎯 Training Scripts

### `quick_train_m4pro.sh`

**Purpose**: One-click training for Mac M4 Pro

**What it does**:
1. ✅ Validates build
2. ✅ Launches TensorBoard in new terminal
3. ✅ Opens browser to TensorBoard UI
4. ✅ Launches live log monitor
5. ✅ Starts training with MPS backend

**Usage**:
```bash
./scripts/quick_train_m4pro.sh
```

**After running**:
- Training runs in current terminal
- TensorBoard accessible at: http://localhost:6006
- Log monitor in separate terminal
- Press Ctrl+C to stop training

---

### `quick_train_production.sh`

**Purpose**: One-click training for production GPUs

**What it does**:
1. ✅ Uses production config (batch_size=32, 480×640 images)
2. ✅ Launches TensorBoard
3. ✅ Opens monitoring dashboard
4. ✅ Starts training with CUDA backend

**Usage**:
```bash
./scripts/quick_train_production.sh
```

**Requirements**:
- NVIDIA GPU with CUDA support
- 40GB+ VRAM recommended

---

### `train_with_monitoring.sh`

**Purpose**: Flexible orchestration script for any config

**What it does**:
1. ✅ Checks build status (builds if needed)
2. ✅ Validates config file
3. ✅ Launches TensorBoard in new terminal
4. ✅ Optionally launches log monitor
5. ✅ Starts training

**Usage**:
```bash
# Basic (uses default config)
./scripts/train_with_monitoring.sh

# Custom config
./scripts/train_with_monitoring.sh configs/train_config.yaml

# Custom config + custom port
./scripts/train_with_monitoring.sh configs/train_config.yaml 8080
```

**Parameters**:
- `$1`: Config file path (default: `configs/train_config_m4pro.yaml`)
- `$2`: TensorBoard port (default: `6006`)

**Interactive prompts**:
- Asks if you want to launch live monitor
- Can skip if you prefer manual monitoring

---

## 📊 Monitoring Scripts

### `watch_training.sh`

**Purpose**: Real-time log monitoring

**What it does**:
- Displays live training logs with color highlighting
- Follows log file (like `tail -f`)
- Highlights epochs, losses, warnings, errors
- Shows metrics updates

**Usage**:
```bash
# Default log file
./scripts/watch_training.sh

# Custom log file
./scripts/watch_training.sh logs/custom_training.log
```

**Features**:
- 🎨 **Color highlighting**: Epochs (blue), warnings (yellow), errors (red)
- 📈 **Metric highlighting**: Loss values and accuracy metrics
- ⏱️ **Real-time**: Updates instantly as logs are written
- 🔄 **Auto-recovery**: Waits if log file doesn't exist yet

**Example output**:
```
👀 Watching Training Logs

Log file: ./logs/training.log

=====================================
--- Epoch 1/100 ---
Training: [100%] Batch 645/645 | Loss: 1.2345
Validation: [100%] Sample 500/500
Val Loss: 1.1234 | abs_rel: 0.1876 | rmse: 0.5432 | δ<1.25: 0.8234
=====================================
```

---

### `monitor_training.py`

**Purpose**: Advanced Python-based log monitor

**Features**:
- 📊 **Dashboard mode**: Auto-refreshing display
- 📈 **Metrics summary**: Shows latest validation metrics
- 🎨 **Syntax highlighting**: ANSI colors for readability
- 🔍 **CSV parsing**: Reads metrics.csv for summaries

**Usage**:
```bash
# Basic follow mode (like tail -f)
python3 scripts/monitor_training.py

# Dashboard mode (auto-refreshing every 5s)
python3 scripts/monitor_training.py --dashboard

# Custom refresh interval
python3 scripts/monitor_training.py --dashboard --refresh 10

# Show last 100 lines
python3 scripts/monitor_training.py --lines 100

# Custom files
python3 scripts/monitor_training.py \
    --log logs/custom.log \
    --metrics logs/custom_metrics.csv
```

**Dashboard mode example**:
```
=================================
📊 TRAINING MONITOR - Real-time Logs
=================================
🕐 2025-01-23 14:32:45
📁 Log file: ./logs/training.log
---------------------------------

📝 RECENT LOGS (last 20 lines):
---------------------------------
[Epoch 15] Training...
[100%] Loss: 0.8234
[Epoch 15] Validation...
Val Loss: 0.7891 | abs_rel: 0.1234

==================================
📈 LATEST METRICS SUMMARY
==================================

  epoch          : 15
  train_loss     : 0.823400
  val_loss       : 0.789100
  abs_rel        : 0.1234
  rmse           : 0.4567
  a1             : 0.8912

🔄 Refreshing every 5s...
```

**Options**:
- `--log FILE`: Training log file (default: `./logs/training.log`)
- `--metrics FILE`: Metrics CSV (default: `./logs/metrics.csv`)
- `--lines N`: Initial lines to show (default: 50)
- `--dashboard`: Enable dashboard mode
- `--refresh N`: Refresh interval in seconds (default: 5)
- `--no-follow`: Print and exit (no follow)

---

## 🌐 TensorBoard Scripts

### `start_tensorboard.sh`

**Purpose**: Standalone TensorBoard launcher

**What it does**:
1. Starts TensorBoard server
2. Opens browser automatically
3. Displays connection info

**Usage**:
```bash
# Default (port 6006, logdir ./runs)
./scripts/start_tensorboard.sh

# Custom port
./scripts/start_tensorboard.sh 8080

# Custom port + logdir
./scripts/start_tensorboard.sh 8080 ./runs/experiment1
```

**Output**:
```
📊 Starting TensorBoard

Port: 6006
Log Directory: ./runs
URL: http://localhost:6006

TensorBoard 2.x.x at http://localhost:6006 (Press CTRL+C to quit)
```

---

### `launch_tensorboard.py`

**Purpose**: Advanced Python TensorBoard launcher

**Features**:
- 🌐 **Auto-opens browser** after 3-second delay
- 🔄 **Fast reload**: 5-second refresh interval
- 🌍 **Network access**: Optional `--bind-all` for remote access
- 📁 **Auto-creates directories**: Creates logdir if missing
- ✅ **Validation**: Checks TensorBoard installation

**Usage**:
```bash
# Basic
python3 scripts/launch_tensorboard.py

# Custom port
python3 scripts/launch_tensorboard.py --port 8080

# Different experiment
python3 scripts/launch_tensorboard.py --logdir ./runs/experiment2

# Allow remote access
python3 scripts/launch_tensorboard.py --bind-all

# Don't open browser
python3 scripts/launch_tensorboard.py --no-browser
```

**Options**:
- `--logdir DIR`: Log directory (default: `./runs`)
- `--port PORT`: Port number (default: `6006`)
- `--no-browser`: Don't open browser automatically
- `--bind-all`: Allow network access (bind to 0.0.0.0)

**Example output**:
```
======================================================================
🚀 Launching TensorBoard
======================================================================
📁 Log Directory: /path/to/project/runs
🌐 Port: 6006
🔗 URL: http://localhost:6006
======================================================================

💡 Tip: Keep this terminal open while training
   Press Ctrl+C to stop TensorBoard

🌐 Opening TensorBoard in browser in 3 seconds...
▶️  Starting TensorBoard...

TensorBoard 2.x.x at http://localhost:6006 (Press CTRL+C to quit)
```

---

## 🎓 Advanced Usage

### Multiple Experiments Comparison

Run multiple experiments in parallel and compare in TensorBoard:

```bash
# Terminal 1: Experiment 1 (standard LR)
./build/train \
    --config configs/train_config_m4pro.yaml \
    --experiment m4pro_standard \
    --tensorboard true

# Terminal 2: Experiment 2 (high LR)
# Edit config to set learning_rate: 3.0e-4
./build/train \
    --config configs/train_config_m4pro_highlr.yaml \
    --experiment m4pro_highlr \
    --tensorboard true

# Terminal 3: View both in TensorBoard
./scripts/start_tensorboard.sh 6006 ./runs
```

In TensorBoard, you'll see both experiments side-by-side!

---

### Remote Training Monitoring

Train on a remote server and monitor from your laptop:

**On server**:
```bash
# Start training
./build/train --config configs/train_config_production.yaml --tensorboard true

# Start TensorBoard with network access
python3 scripts/launch_tensorboard.py --bind-all --port 6006
```

**On laptop**:
```bash
# SSH tunnel
ssh -L 6006:localhost:6006 user@server

# Open browser to http://localhost:6006
```

---

### Automated Training Pipeline

Create a script for overnight training:

```bash
#!/bin/bash
# overnight_training.sh

# Run 3 experiments sequentially
experiments=(
    "configs/train_config_baseline.yaml"
    "configs/train_config_highlr.yaml"
    "configs/train_config_highreproj.yaml"
)

for config in "${experiments[@]}"; do
    echo "Starting: $config"
    ./build/train --config "$config" --tensorboard true
    echo "Completed: $config"
done

echo "All experiments complete!"
```

---

### Custom Monitoring Dashboard

Create a custom dashboard with multiple monitors:

**Terminal 1**: TensorBoard
```bash
./scripts/start_tensorboard.sh 6006
```

**Terminal 2**: Live logs
```bash
python3 scripts/monitor_training.py --dashboard --refresh 3
```

**Terminal 3**: GPU monitor (Linux)
```bash
watch -n 1 nvidia-smi
```

**Terminal 4**: System resources (Mac)
```bash
# Install htop: brew install htop
htop
```

**Terminal 5**: Training
```bash
./build/train --config configs/train_config_m4pro.yaml --tensorboard true
```

---

## 🐛 Troubleshooting

### "TensorBoard not found"

**Problem**: Python script can't find TensorBoard

**Solution**:
```bash
# Install TensorBoard
pip install tensorboard

# Or with conda
conda install tensorboard

# Verify installation
tensorboard --version
```

---

### "Permission denied" when running scripts

**Problem**: Scripts not executable

**Solution**:
```bash
# Make scripts executable
chmod +x scripts/*.sh scripts/*.py

# Or individually
chmod +x scripts/quick_train_m4pro.sh
```

---

### TensorBoard shows "No data"

**Problem**: TensorBoard running but shows empty dashboard

**Solutions**:

1. **Check logdir**: Make sure TensorBoard is pointing to correct directory
   ```bash
   ls ./runs/
   # Should show subdirectories with event files
   ```

2. **Wait for first epoch**: TensorBoard needs some data to display
   ```bash
   # Check if training has started logging
   ls ./runs/*/events*
   ```

3. **Refresh browser**: Force reload in browser (Cmd+Shift+R or Ctrl+F5)

---

### "Cannot open new terminal" error

**Problem**: Script can't open terminal windows (macOS/Linux)

**Solution**: Run components manually in separate terminals:

**Terminal 1**:
```bash
./scripts/start_tensorboard.sh 6006
```

**Terminal 2**:
```bash
./scripts/watch_training.sh
```

**Terminal 3**:
```bash
./build/train --config configs/train_config_m4pro.yaml --tensorboard true
```

---

### Monitor shows old logs

**Problem**: Monitor displaying cached/old log data

**Solution**:
```bash
# Clear old logs
rm logs/*.log logs/*.csv

# Restart training
./scripts/quick_train_m4pro.sh
```

---

### Port already in use

**Problem**: TensorBoard can't start on port 6006

**Solution**:
```bash
# Find process using port
lsof -i :6006

# Kill process
kill -9 <PID>

# Or use different port
./scripts/start_tensorboard.sh 6007
```

---

## 📚 Tips and Best Practices

### 1. Use `screen` or `tmux` for Long Training

For training that runs overnight:

```bash
# Start screen session
screen -S training

# Run training
./scripts/quick_train_m4pro.sh

# Detach: Ctrl+A then D
# Reattach later: screen -r training
```

### 2. Disable Auto-Sleep (Mac)

```bash
# Prevent sleep while training
caffeinate -d &

# Run training
./scripts/quick_train_m4pro.sh

# Kill caffeinate when done
killall caffeinate
```

### 3. Bookmark TensorBoard

Add to your browser bookmarks:
- http://localhost:6006 - TensorBoard

### 4. Log Rotation

For very long training:

```bash
# Archive old logs before new training
mkdir -p logs/archive
mv logs/*.log logs/*.csv logs/archive/
```

### 5. Quick Checks

```bash
# Check if training is running
ps aux | grep train

# Check TensorBoard status
lsof -i :6006

# Check latest metrics
tail -n 5 logs/metrics.csv

# Check GPU usage (Mac)
sudo powermetrics --samplers gpu_power -i 5000
```

---

## 🎯 Recommended Workflow

### For Mac M4 Pro:

```bash
# 1. One command to start everything
./scripts/quick_train_m4pro.sh

# 2. Open TensorBoard in browser (auto-opens)
#    http://localhost:6006

# 3. Watch progress in TensorBoard UI

# 4. Check detailed logs if needed
#    Already displayed in separate terminal

# 5. Let it train!
```

### For Production:

```bash
# 1. Start in screen session
screen -S production_training

# 2. Launch orchestrated training
./scripts/quick_train_production.sh

# 3. Detach and logout
#    Ctrl+A, then D

# 4. Monitor remotely via TensorBoard
#    SSH tunnel: ssh -L 6006:localhost:6006 user@server
```

---

## 📖 See Also

- [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) - Complete training guide
- [MAC_M4_PRO_GUIDE.md](../MAC_M4_PRO_GUIDE.md) - Mac-specific optimization
- [README.md](../README.md) - Project overview

---

**Version**: 1.0.0
**Last Updated**: 2025-01-XX
**Maintainer**: Camera-Aware Depth Estimation Team
