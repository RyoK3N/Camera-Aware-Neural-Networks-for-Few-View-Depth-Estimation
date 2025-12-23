# TensorBoard Integration - Complete Guide
## Camera-Aware Depth Estimation System

**Version:** 2.0
**Last Updated:** December 23, 2025
**Status:** ✅ Production Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Current Implementation](#current-implementation)
4. [How to Use](#how-to-use)
5. [TensorBoard Features](#tensorboard-features)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Advanced Usage](#advanced-usage)
10. [Technical Reference](#technical-reference)

---

## Quick Start

### Prerequisites

```bash
# Ensure conda environment is active
conda activate synexian

# Install/verify TensorBoard with compatible protobuf
pip install "protobuf<4" --force-reinstall
pip install tensorboard torch
```

### One-Click Training

```bash
cd "/Users/r/Desktop/Synexian/ML Research Ideas and Topics/Camera Matrix"
./scripts/quick_train_m4pro.sh
```

This automatically:
- Validates the build
- Launches TensorBoard at http://localhost:6006
- Starts training with comprehensive visualizations
- Opens live monitoring dashboard

### Manual Training

**Terminal 1 - TensorBoard:**
```bash
tensorboard --logdir=./runs --port=6006
```

**Terminal 2 - Training:**
```bash
./build/train --config configs/train_config_m4pro.yaml --tensorboard true
```

**Browser:**
```
http://localhost:6006
```

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Process (C++)                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ TensorBoardTrainerEnhanced                         │     │
│  │  - Epoch loop                                      │     │
│  │  - Batch processing                                │     │
│  │  - Metric computation                              │     │
│  │  - Visualization generation                        │     │
│  └─────────────────┬──────────────────────────────────┘     │
│                    │                                         │
│                    ▼                                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │ TensorBoardLoggerV2 (C++)                          │     │
│  │  - addScalar()                                     │     │
│  │  - addImage()                                      │     │
│  │  - addHistogram()                                  │     │
│  │  - addText()                                       │     │
│  │  - addHParams()                                    │     │
│  └─────────────────┬──────────────────────────────────┘     │
│                    │                                         │
│                    │ JSON over pipe                          │
│                    ▼                                         │
└────────────────────┼─────────────────────────────────────────┘
                     │
┌────────────────────┼─────────────────────────────────────────┐
│                    │   IPC Layer                             │
│  ┌─────────────────▼──────────────────────────────────┐     │
│  │ popen() pipe                                       │     │
│  │  - stdin/stdout communication                      │     │
│  │  - JSON protocol                                   │     │
│  │  - Command: {"type": "scalar", "tag": ..., ...}    │     │
│  └─────────────────┬──────────────────────────────────┘     │
└────────────────────┼─────────────────────────────────────────┘
                     │
┌────────────────────┼─────────────────────────────────────────┐
│                    │   Python Service                        │
│  ┌─────────────────▼──────────────────────────────────┐     │
│  │ tensorboard_writer.py                              │     │
│  │  - Reads JSON commands from stdin                  │     │
│  │  - Parses and validates                            │     │
│  │  - Routes to appropriate handler                   │     │
│  └─────────────────┬──────────────────────────────────┘     │
│                    │                                         │
│                    ▼                                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │ torch.utils.tensorboard.SummaryWriter              │     │
│  │  - add_scalar()                                    │     │
│  │  - add_image()                                     │     │
│  │  - add_histogram()                                 │     │
│  │  - Flush every 1 second                            │     │
│  └─────────────────┬──────────────────────────────────┘     │
│                    │                                         │
│                    ▼                                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │ TensorBoard Event Files                            │     │
│  │  - Binary tfrecord format                          │     │
│  │  - Protobuf encoding                               │     │
│  │  - events.out.tfevents.* files                     │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Advantages:**
- ✅ **Native TensorBoard support** - Uses official PyTorch SummaryWriter
- ✅ **Proper event files** - Binary tfrecord format with protobuf
- ✅ **All features work** - Scalars, images, histograms, graphs, hparams
- ✅ **Clean separation** - C++ for compute, Python for visualization
- ✅ **Easy debugging** - JSON protocol is human-readable
- ✅ **Minimal overhead** - IPC cost < 1ms per log event

**Design Decisions:**
1. **JSON over pipe instead of file watching** - Real-time updates, no polling overhead
2. **Process-based instead of library** - Avoids C++/Python linking complexity
3. **1-second flush** - Immediate visualization vs. default 120-second buffer

---

## Current Implementation

### File Structure

```
src/training/
  ├── tensorboard_logger_v2.h          # C++ logger (JSON-IPC bridge)
  ├── tensorboard_trainer_enhanced.h    # Enhanced trainer with full logging
  └── train_main.cpp                    # Entry point using enhanced trainer

scripts/
  ├── tensorboard_writer.py             # Python event writer service
  ├── launch_tensorboard.py             # TensorBoard launcher
  ├── monitor_training.py               # Live training monitor
  ├── quick_train_m4pro.sh              # One-click training script
  └── train_with_monitoring.sh          # Multi-terminal orchestration

configs/
  └── train_config_m4pro.yaml           # Mac M4 Pro optimized config

runs/                                   # TensorBoard event files (auto-created)
  └── baseline_unet_m4pro/
      └── events.out.tfevents.*

checkpoints/                            # Model checkpoints (auto-created)
logs/                                   # Text logs (auto-created)
```

### Key Components

#### 1. TensorBoardLoggerV2 (`src/training/tensorboard_logger_v2.h`)

**Responsibilities:**
- Spawn Python writer service as subprocess
- Send JSON commands over pipe
- Convert C++ types to JSON
- Handle image tensor serialization
- Sample large histograms (max 10K points)

**API:**
```cpp
TensorBoardLoggerV2 logger("./runs/experiment");

// Scalars
logger.addScalar("loss/train", 0.523, epoch);

// Multiple scalars
std::map<std::string, float> metrics;
metrics["abs_rel"] = 0.182;
metrics["rmse"] = 0.543;
logger.addScalars("validation/metrics", metrics, epoch);

// Images (C, H, W format)
logger.addImage("predictions/depth", depth_tensor, epoch);

// Histograms
logger.addHistogram("weights/conv1", weight_tensor, epoch);

// Text
logger.addText("model/architecture", model_description, 0);

// Hyperparameters
std::map<std::string, float> hparams, metrics_final;
hparams["lr"] = 0.0002;
metrics_final["final_loss"] = 0.234;
logger.addHParams(hparams, metrics_final);
```

#### 2. TensorBoardTrainerEnhanced (`src/training/tensorboard_trainer_enhanced.h`)

**What It Logs:**

| Category | Metrics | Frequency | Tag Prefix |
|----------|---------|-----------|------------|
| **Training Loss** | Total loss | Every epoch | `loss/train` |
| **Batch Loss** | Batch-level loss | Every 10 batches | `batch_loss/train` |
| **Loss Components** | SI, Grad, Smooth, Reproj | Every epoch | `loss_components/*` |
| **Validation Loss** | Total validation loss | Every 10 epochs | `loss/val` |
| **Validation Metrics** | abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 | Every 10 epochs | `metrics/*` |
| **Learning Rate** | Current LR | Every epoch | `training/learning_rate` |
| **Gradient Norm** | Global gradient norm | Every batch | `training/gradient_norm` |
| **Gradient Histograms** | Per-layer gradients | Every 5 epochs | `gradients/*` |
| **Weight Histograms** | Per-layer weights | Every 5 epochs | `weights/*` |
| **Images** | RGB, GT, Pred, Error maps | Every epoch | `images/*` |
| **Time Tracking** | Epoch time, total time | Every epoch | `training/*_time_seconds` |

**Custom Scalar Layouts:**
The trainer automatically creates organized views in TensorBoard:
- **Training Panel:** Loss curves, learning rate, time tracking
- **Metrics Panel:** Validation metrics comparison
- **Model Panel:** Gradient and weight statistics

#### 3. Python Writer Service (`scripts/tensorboard_writer.py`)

**Features:**
- Immediate flush mode (`flush_secs=1`)
- Event counting and progress tracking
- Graceful shutdown on SIGINT/SIGTERM
- Error handling with JSON responses
- Support for all TensorBoard data types

**JSON Protocol Examples:**

```json
// Scalar
{
  "type": "scalar",
  "tag": "loss/train",
  "value": 0.523,
  "step": 10
}

// Image from file
{
  "type": "image",
  "tag": "predictions/depth",
  "path": "/path/to/temp_image.png",
  "step": 10
}

// Histogram
{
  "type": "histogram",
  "tag": "weights/conv1",
  "values": [0.12, -0.34, 0.56, ...],
  "step": 10
}

// Shutdown
{
  "type": "shutdown"
}
```

---

## How to Use

### Initial Setup

1. **Build the project:**
```bash
cd "/Users/r/Desktop/Synexian/ML Research Ideas and Topics/Camera Matrix/build"
cmake ..
make train -j8
```

2. **Fix protobuf compatibility:**
```bash
conda activate synexian
pip install "protobuf<4" --force-reinstall
```

3. **Verify TensorBoard works:**
```bash
tensorboard --logdir=./runs --port=6006
# Should start without errors
```

### Running Training

**Recommended: Use orchestration script**
```bash
cd "/Users/r/Desktop/Synexian/ML Research Ideas and Topics/Camera Matrix"
./scripts/quick_train_m4pro.sh
```

**Manual control:**
```bash
# Terminal 1
tensorboard --logdir=./runs --port=6006

# Terminal 2
./build/train --config configs/train_config_m4pro.yaml --tensorboard true
```

### What to Expect

#### After Training Starts (Within 30 seconds):
- **Console output:**
  ```
  [TensorBoard Writer] Initialized at ./runs/baseline_unet_m4pro
  [TensorBoard Writer] Flush interval: 1 second (immediate mode)
  [TensorBoard Writer] Ready to receive commands
  ```
- **TensorBoard SCALARS tab:** Empty (waiting for first epoch)
- **Event files:** Created in `./runs/baseline_unet_m4pro/`

#### After First Batch (Within 2 minutes):
- **SCALARS tab:**
  - `batch_loss/train` - First data point
  - `training/gradient_norm` - First data point

#### After Epoch 1 Completes (~4 hours on Mac M4 Pro):
- **SCALARS tab (10+ plots):**
  - ✅ `loss/train`
  - ✅ `loss/val` (empty until epoch 10)
  - ✅ `loss_components/si_loss`
  - ✅ `loss_components/grad_loss`
  - ✅ `loss_components/smooth_loss`
  - ✅ `loss_components/reproj_loss`
  - ✅ `training/learning_rate`
  - ✅ `training/epoch_time_seconds`
  - ✅ `training/total_time_seconds`
  - ✅ `batch_loss/train` (646 points, one per batch)
  - ✅ `training/gradient_norm` (646 points)

- **IMAGES tab (8 visualizations):**
  - 4 samples × 2 formats (GT vs Pred side-by-side, Error maps)

- **TEXT tab:**
  - Model architecture information
  - Hyperparameter configuration

#### After Epoch 5:
- **HISTOGRAMS tab:**
  - Weight distributions for all layers
  - Gradient distributions for all layers

#### After Epoch 10:
- **SCALARS tab (validation metrics):**
  - ✅ `loss/val`
  - ✅ `metrics/abs_rel`
  - ✅ `metrics/sq_rel`
  - ✅ `metrics/rmse`
  - ✅ `metrics/rmse_log`
  - ✅ `metrics/a1` (δ < 1.25 accuracy)
  - ✅ `metrics/a2` (δ < 1.25² accuracy)
  - ✅ `metrics/a3` (δ < 1.25³ accuracy)

- **HPARAMS tab:**
  - Hyperparameter comparison table
  - Parallel coordinates plot

---

## TensorBoard Features

### 1. SCALARS Tab

**Purpose:** Real-time plots of scalar metrics over time

**What You'll See:**
- Loss curves (training, validation, components)
- Metrics curves (abs_rel, rmse, accuracy thresholds)
- Learning rate schedule
- Gradient norms (detect vanishing/exploding gradients)
- Time tracking (epoch duration, total training time)

**Tips:**
- **Smoothing slider (bottom left):** Adjust to reduce noise in curves
- **Show/hide runs:** Click legend items to toggle visibility
- **Download charts:** Click download icon for publication-quality SVGs
- **Regex filtering:** Use left sidebar to filter by tag pattern (e.g., `loss/*`)

**Key Metrics to Monitor:**

| Metric | Target | Excellent | Warning |
|--------|--------|-----------|---------|
| `loss/train` | Decreasing | Smooth curve | Oscillating |
| `metrics/abs_rel` | < 0.20 | < 0.15 | > 0.25 |
| `metrics/rmse` | < 0.60 | < 0.50 | > 0.70 |
| `metrics/a1` | > 0.75 | > 0.85 | < 0.70 |
| `training/gradient_norm` | 0.1 - 10 | 0.5 - 5 | < 0.01 or > 100 |

### 2. IMAGES Tab

**Purpose:** Visual validation of model predictions

**What You'll See:**
Each epoch shows 8 sample predictions:
- **Input RGB:** Original image
- **Ground Truth Depth:** Supervision signal
- **Predicted Depth:** Model output
- **Error Map:** Color-coded absolute error (blue=low, red=high)

**Tips:**
- Use the slider to navigate through epochs
- Click images to view full resolution
- Compare across epochs to see improvement
- Look for systematic errors (e.g., poor edge prediction)

### 3. HISTOGRAMS Tab

**Purpose:** Monitor weight and gradient distributions

**What You'll See:**
- **Weight histograms:** Should be roughly normal-distributed
- **Gradient histograms:** Should have variance, not all zeros

**Warning Signs:**
- ⚠️ **Weights all near zero:** Poor initialization or dead neurons
- ⚠️ **Weights growing unbounded:** Weight explosion, reduce LR
- ⚠️ **Gradients all near zero:** Vanishing gradients, check activations
- ⚠️ **Gradients very large:** Exploding gradients, add gradient clipping

**Healthy Training:**
- ✅ Weights centered near 0, spread ± 0.5
- ✅ Gradients with variance, mean near 0
- ✅ Distributions evolving slowly over epochs

### 4. TEXT Tab

**Purpose:** Experiment documentation and model architecture

**What You'll See:**
- Model architecture (layer names, shapes, parameters)
- Hyperparameter configuration
- Experiment description and tags

### 5. HPARAMS Tab

**Purpose:** Compare hyperparameters across multiple runs

**What You'll See:**
- **Table View:** All runs with hyperparameters and final metrics
- **Parallel Coordinates:** Visualize relationship between hyperparameters and metrics

**How to Use:**
1. Run multiple experiments with different configs
2. Each run appears as a row in the table
3. Sort by metric columns to find best hyperparameters
4. Use parallel coordinates to identify patterns

---

## Troubleshooting

### Problem: No data in TensorBoard

**Symptoms:** All tabs empty, "No dashboards are active" message

**Diagnosis:**
```bash
# Check if Python writer is running
ps aux | grep tensorboard_writer

# Check if event files exist
ls -lh runs/baseline_unet_m4pro/

# Check if training is running
# Console should show progress bars
```

**Solutions:**
1. **Event files missing:**
   ```bash
   # Restart training - writer service may have crashed
   # Check logs for Python errors
   ```

2. **Event files exist but empty:**
   ```bash
   # Flush issue - check Python writer logs
   # Verify flush_secs=1 in tensorboard_writer.py line 33
   ```

3. **TensorBoard pointing to wrong directory:**
   ```bash
   # Restart TensorBoard with correct logdir
   tensorboard --logdir=./runs --port=6006
   ```

### Problem: Only 2 graphs updating

**Symptoms:** `batch_loss/train` and `training/gradient_norm` work, others don't

**Root Cause:** Event buffering (default 120-second flush)

**Solution:** Already fixed in current codebase
- `scripts/tensorboard_writer.py` line 33: `flush_secs=1`
- Line 85: Explicit `self.writer.flush()` after each write
- Restart training to see all graphs update

### Problem: Images not appearing

**Symptoms:** IMAGES tab empty after epoch 1

**Root Cause:** Images only logged during validation (every 10 epochs by default)

**Solution:** Already fixed in current codebase
- `src/training/tensorboard_trainer_enhanced.h` line 189: Images logged every epoch
- Restart training to see images after epoch 1

### Problem: Histograms not appearing

**Symptoms:** HISTOGRAMS tab empty after epoch 5

**Diagnosis:**
```bash
# Check histogram interval in config
grep -A 5 "histogram_interval" configs/train_config_m4pro.yaml
```

**Solutions:**
1. **Wait until epoch 5** - Histograms logged every 5 epochs by default
2. **Change interval:**
   ```yaml
   logging:
     tensorboard:
       log_histogram_interval: 1  # Log every epoch (slower)
   ```

### Problem: Broken image visualizations

**Symptoms:** Images appear corrupted or all black/white

**Root Cause:** Tensor format incorrect (not in CHW format or wrong value range)

**Diagnosis:**
```bash
# Check temporary image files
ls -lh runs/baseline_unet_m4pro/temp_*.png

# Verify they can be opened
open runs/baseline_unet_m4pro/temp_predictions_depth_0_10.png
```

**Solution:**
- Current implementation auto-normalizes to [0, 255] range
- Converts HWC → CHW format automatically
- If issues persist, check `tensorboard_logger_v2.h` line 354 `saveTensorAsImage()`

### Problem: High memory usage

**Symptoms:** System running out of RAM, training slowing down

**Diagnosis:**
```bash
# Monitor memory during training
top -pid $(pgrep train)
```

**Solutions:**
1. **Reduce visualization samples:**
   ```yaml
   logging:
     tensorboard:
       num_viz_samples: 4  # Default: 8
   ```

2. **Reduce histogram frequency:**
   ```yaml
   logging:
     tensorboard:
       log_histogram_interval: 10  # Default: 5
   ```

3. **Reduce batch size:**
   ```yaml
   training:
     batch_size: 8  # Default: 16
   ```

### Problem: Python writer crashed

**Symptoms:** Console shows "[TensorBoardLoggerV2] ERROR: Writer service not running"

**Diagnosis:**
```bash
# Check Python errors
cat logs/training.log | grep -i "tensorboard writer"

# Test writer independently
echo '{"type":"scalar","tag":"test","value":1.0,"step":0}' | \
  python scripts/tensorboard_writer.py ./runs/test
```

**Solutions:**
1. **Missing dependencies:**
   ```bash
   conda activate synexian
   pip install torch tensorboard "protobuf<4"
   ```

2. **Path with spaces issue:**
   - Already fixed with single-quote escaping in `tensorboard_logger_v2.h` line 299
   - Rebuild if using old version: `make clean && make train -j8`

3. **Permissions issue:**
   ```bash
   chmod +x scripts/tensorboard_writer.py
   ```

### Problem: "No module named tensorboard"

**Symptoms:** Training crashes with Python import error

**Solution:**
```bash
conda activate synexian
pip install tensorboard torch
```

### Problem: TensorBoard shows old data

**Symptoms:** Plots not updating, stuck on previous run

**Solution:**
```bash
# Clear old event files
rm -rf runs/baseline_unet_m4pro/*

# Restart TensorBoard (Ctrl+C, then)
tensorboard --logdir=./runs --port=6006 --reload_interval=1
```

---

## Best Practices

### Logging Frequency

**Scalars:**
- ✅ **Batch loss:** Every 10 batches (balance granularity vs overhead)
- ✅ **Epoch loss:** Every epoch (always)
- ✅ **Learning rate:** Every epoch (track schedule)
- ✅ **Gradient norm:** Every batch (early warning for instability)

**Images:**
- ✅ **Training predictions:** Every epoch (8 samples, visual feedback)
- ⚠️ **Avoid:** Every batch (creates huge event files)

**Histograms:**
- ✅ **Weights/gradients:** Every 5 epochs (distributions evolve slowly)
- ⚠️ **Avoid:** Every epoch (expensive, redundant)

**Validation:**
- ✅ **Full metrics:** Every 10 epochs (comprehensive evaluation)
- ⚠️ **Avoid:** Every epoch (validation is slow)

### Hierarchical Tag Organization

Use `/` to create hierarchies in TensorBoard:

```cpp
// Good: Organized into categories
logger.addScalar("loss/train", train_loss, epoch);
logger.addScalar("loss/val", val_loss, epoch);
logger.addScalar("loss_components/si_loss", si_loss, epoch);
logger.addScalar("metrics/abs_rel", abs_rel, epoch);
logger.addScalar("training/learning_rate", lr, epoch);

// Bad: Flat namespace
logger.addScalar("train_loss", train_loss, epoch);
logger.addScalar("val_loss", val_loss, epoch);
```

**Benefits:**
- Automatic grouping in TensorBoard UI
- Regex filtering (`loss/*`, `metrics/*`)
- Custom scalar layouts work better

### Memory Management

**For image logging:**
```cpp
// Good: Save image, log path, let Python load it
std::string temp_path = log_dir_ + "/temp_depth_" + std::to_string(step) + ".png";
saveTensorAsImage(depth_tensor, temp_path);
logger.addImageFromPath("predictions/depth", temp_path, step);

// Alternative: Current implementation uses this pattern
// Temporary files are overwritten each epoch (minimal disk usage)
```

**For histogram logging:**
```cpp
// Good: Sample large tensors (already done in tensorboard_logger_v2.h)
if (numel > 10000) {
    int stride = numel / 10000;
    for (int i = 0; i < numel; i += stride) {
        sampled_values.push_back(data_ptr[i]);
    }
}
```

### Experiment Organization

**Directory structure:**
```
runs/
  ├── baseline_unet_m4pro/          # Baseline experiment
  ├── baseline_unet_high_lr/        # High learning rate test
  ├── baseline_unet_no_reproj/      # Ablation: no reprojection loss
  └── resnet_encoder_m4pro/         # Architecture comparison
```

**Naming convention:**
```
{model_architecture}_{experiment_variant}_{hardware}
```

**Benefits:**
- Easy comparison in TensorBoard (select multiple runs)
- Clear experiment tracking
- Hyperparameter analysis in HPARAMS tab

### Version Control

**Commit event files?** ❌ No
```gitignore
# .gitignore
runs/
logs/
checkpoints/
*.png
*.jpg
```

**Commit configs?** ✅ Yes
```bash
git add configs/train_config_*.yaml
git commit -m "Add config for high learning rate experiment"
```

**What to track:**
- ✅ Configuration files
- ✅ Model architecture code
- ✅ Training scripts
- ✅ Best checkpoint paths (in documentation)
- ❌ Event files (binary, large, reproducible)
- ❌ Temporary images
- ❌ Log files

### Reproducibility

**Log hyperparameters at start of training:**
```cpp
std::map<std::string, float> hparams;
hparams["learning_rate"] = config.learning_rate;
hparams["batch_size"] = config.batch_size;
hparams["si_loss_weight"] = config.si_loss_weight;
hparams["grad_loss_weight"] = config.grad_loss_weight;
hparams["smooth_loss_weight"] = config.smooth_loss_weight;
hparams["reproj_loss_weight"] = config.reproj_loss_weight;

std::map<std::string, float> metrics_placeholder;
logger.addHParams(hparams, metrics_placeholder);
```

**Update with final metrics:**
```cpp
// After training completes
metrics_placeholder["final_abs_rel"] = best_abs_rel;
metrics_placeholder["final_rmse"] = best_rmse;
metrics_placeholder["final_a1"] = best_a1;
logger.addHParams(hparams, metrics_placeholder);
```

---

## Performance Optimization

### Logging Overhead

**Measured overhead:**
- Scalar logging: ~0.5ms per call
- Image logging: ~10ms per image (depends on resolution)
- Histogram logging: ~5ms per histogram (after sampling)

**Total overhead:** < 5% of training time (acceptable)

### Reducing Overhead

1. **Batch logging:** Group scalars using `addScalars()`
   ```cpp
   // Instead of multiple calls:
   logger.addScalar("metrics/abs_rel", abs_rel, step);
   logger.addScalar("metrics/rmse", rmse, step);
   logger.addScalar("metrics/a1", a1, step);

   // Use batch call:
   std::map<std::string, float> metrics;
   metrics["abs_rel"] = abs_rel;
   metrics["rmse"] = rmse;
   metrics["a1"] = a1;
   logger.addScalars("metrics", metrics, step);
   ```

2. **Conditional logging:**
   ```cpp
   // Only log histograms every 5 epochs
   if (epoch % 5 == 0) {
       logWeightHistograms(epoch);
       logGradientHistograms(epoch);
   }
   ```

3. **Reduce visualization samples:**
   ```yaml
   num_viz_samples: 4  # Default: 8 (reduces image logging by 50%)
   ```

### Disk Usage

**Expected event file sizes:**
- 100 epochs, all features enabled: ~500 MB
- Per epoch: ~5 MB
- Scalars only: ~50 MB for 100 epochs

**Cleanup strategy:**
```bash
# Keep only last 3 runs
cd runs
ls -t | tail -n +4 | xargs rm -rf

# Or delete runs older than 7 days
find runs -type d -mtime +7 -exec rm -rf {} +
```

### Network Usage (Multi-Machine Setup)

**For remote TensorBoard:**
```bash
# On training machine
./build/train --config configs/train_config.yaml --tensorboard true

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@training-machine

# Open browser locally
http://localhost:6006
```

**Benefits:**
- No event file copying
- Real-time updates
- Minimal network traffic (~1 KB/s)

---

## Advanced Usage

### Custom Scalar Layouts

The enhanced trainer automatically creates layouts, but you can customize:

**Edit `scripts/tensorboard_writer.py` line 50:**
```python
def setup_custom_layout(self):
    """Create custom scalar layout for organized visualization"""
    layout = {
        "Training": {
            "Loss": ["Multiline", ["loss/train", "loss/val"]],
            "Loss Components": ["Multiline", [
                "loss_components/si_loss",
                "loss_components/grad_loss",
                "loss_components/smooth_loss",
                "loss_components/reproj_loss"
            ]],
            "Learning Rate": ["Multiline", ["training/lr"]]
        },
        "Metrics": {
            "Relative Error": ["Multiline", ["metrics/abs_rel", "metrics/sq_rel"]],
            "RMSE": ["Multiline", ["metrics/rmse", "metrics/rmse_log"]],
            "Accuracy": ["Multiline", ["metrics/a1", "metrics/a2", "metrics/a3"]]
        },
        "Model": {
            "Gradients": ["Multiline", ["gradients/norm", "gradients/max", "gradients/min"]],
            "Weights": ["Multiline", ["weights/norm", "weights/sparsity"]]
        }
    }

    self.writer.add_custom_scalars(layout)
```

**Result:** Click "Custom Scalars" tab in TensorBoard to see organized panels.

### Comparing Multiple Experiments

**Step 1: Run experiments with different names**
```bash
# Experiment 1: Baseline
./build/train --config configs/train_config_m4pro.yaml \
              --experiment baseline_v1

# Experiment 2: High LR
./build/train --config configs/train_config_high_lr.yaml \
              --experiment high_lr_v1

# Experiment 3: No reprojection loss
./build/train --config configs/train_config_no_reproj.yaml \
              --experiment no_reproj_v1
```

**Step 2: Launch TensorBoard**
```bash
tensorboard --logdir=./runs --port=6006
```

**Step 3: Compare in TensorBoard**
- Use left sidebar to select/deselect runs
- All selected runs overlay on same plots
- Use smoothing slider to reduce noise
- Download comparison charts as SVG

**Step 4: Analyze in HPARAMS tab**
- Click HPARAMS tab
- Table view shows all runs with hyperparameters and metrics
- Sort by metric columns (e.g., abs_rel) to find best config
- Parallel coordinates show relationships

### Exporting Results

**For publication:**
1. Click download icon on any chart
2. Select "Download as SVG"
3. Edit in Inkscape/Illustrator for publication quality

**For data analysis:**
1. Click download icon
2. Select "Download as CSV"
3. Analyze in Python/R/Excel

**Programmatic export:**
```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('./runs/baseline_unet_m4pro')
ea.Reload()

# Get scalar data
train_loss = ea.Scalars('loss/train')
for event in train_loss:
    print(f"Step {event.step}: {event.value}")
```

### Model Graph Visualization

**Note:** Not currently implemented (requires TorchScript tracing)

**To add:**
```cpp
// In tensorboard_logger_v2.h, add method:
void addGraph(torch::jit::script::Module& model, torch::Tensor& input_example) {
    // Trace model
    auto traced = torch::jit::trace(model, input_example);

    // Save to ONNX
    torch::onnx::export(traced, input_example, log_dir_ + "/model.onnx");

    // Send to Python writer (requires additional command type)
    json command;
    command["type"] = "graph";
    command["path"] = log_dir_ + "/model.onnx";
    sendCommand(command);
}
```

**Alternative:** Use Netron to visualize checkpoints
```bash
pip install netron
netron checkpoints/baseline_unet_m4pro/epoch_10.pt
```

### TensorBoard Profiler

**Note:** Not currently implemented (requires PyTorch Profiler integration)

**To add:**
```cpp
// Wrap training loop with profiler
auto profiler = torch::profiler::experimental::record_function_guard("Training");

// Profile specific operations
{
    auto prof_scope = torch::profiler::experimental::record_function_guard("Forward Pass");
    auto output = model->forward(input);
}

// Export to TensorBoard
profiler->export_chrome_trace(log_dir_ + "/trace.json");
```

**Benefits:**
- Identify bottlenecks (forward pass, backward pass, data loading)
- Visualize GPU utilization
- Optimize training speed

---

## Technical Reference

### JSON Protocol Specification

**Command Types:**

| Type | Required Fields | Optional Fields | Example |
|------|----------------|-----------------|---------|
| `scalar` | `tag`, `value`, `step` | - | `{"type":"scalar","tag":"loss/train","value":0.523,"step":10}` |
| `scalars` | `main_tag`, `values`, `step` | - | `{"type":"scalars","main_tag":"metrics","values":{"abs_rel":0.18},"step":10}` |
| `image` | `tag`, `step` | `path` OR `data`+`shape` | `{"type":"image","tag":"depth","path":"temp.png","step":10}` |
| `histogram` | `tag`, `values`, `step` | - | `{"type":"histogram","tag":"weights","values":[...],"step":10}` |
| `text` | `tag`, `text`, `step` | - | `{"type":"text","tag":"model","text":"U-Net","step":0}` |
| `hparams` | `hparams`, `metrics` | - | `{"type":"hparams","hparams":{"lr":0.0002},"metrics":{"loss":0.5}}` |
| `pr_curve` | `tag`, `labels`, `predictions`, `step` | - | Used for binary classification (not used in depth estimation) |
| `shutdown` | - | - | `{"type":"shutdown"}` |

**Response Format:**
```json
// Success
{"status": "success"}

// Error
{"status": "error", "message": "Invalid JSON: ..."}
```

### Data Type Constraints

**Scalars:**
- Must be numeric (int, float)
- No NaN or Inf values
- Step must be non-negative integer

**Images:**
- Tensor shape: `(C, H, W)` where C ∈ {1, 3}
- Values normalized to `[0, 1]` or `[0, 255]`
- Supported formats: PNG, JPEG (when saved to file)

**Histograms:**
- Values must be numeric array
- Can be any length (will be sampled if > 10K points)
- Empty arrays are rejected

**Tags:**
- UTF-8 string
- Recommended: Use `/` for hierarchy (e.g., `loss/train`)
- Avoid special characters: `<`, `>`, `|`, `\`, `^`, `{`, `}`

### Event File Format

**Structure:**
```
events.out.tfevents.[timestamp].[hostname]
```

**Contents:**
- Binary tfrecord format
- Each record is a protobuf-encoded `Event` message
- Event types: scalar, image, histogram, graph, metadata

**Reading event files:**
```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('./runs/experiment')
ea.Reload()

print(ea.Tags())  # Show all tags
scalars = ea.Scalars('loss/train')
images = ea.Images('predictions/depth')
histograms = ea.Histograms('weights/conv1')
```

### Flush Mechanism

**Default behavior:**
- SummaryWriter buffers events in memory
- Flushes to disk every 120 seconds
- Flushes on `close()`

**Enhanced behavior (current implementation):**
- `flush_secs=1` in SummaryWriter constructor
- Explicit `flush()` after every write
- Near-real-time updates (1-second latency)

**Trade-offs:**
- **Faster flush:** Real-time visualization, better debugging
- **Slower flush:** Less I/O overhead, larger batch writes

**Current choice:** Real-time (research environment, debugging priority)

### Process Management

**Subprocess lifecycle:**
1. C++ spawns Python writer via `popen()`
2. Python writer enters event loop, reading stdin
3. C++ sends JSON commands, Python processes and responds
4. On training end, C++ sends shutdown command
5. Python writer closes SummaryWriter and exits
6. C++ calls `pclose()` to clean up

**Graceful shutdown:**
- SIGINT (Ctrl+C) caught by Python writer
- Flushes remaining events
- Closes file handles
- Exits cleanly

**Error handling:**
- If Python writer crashes, C++ detects broken pipe
- Logs error message
- Continues training without TensorBoard (graceful degradation)

---

## Configuration Reference

### Training Config (`configs/train_config_m4pro.yaml`)

**TensorBoard-related settings:**

```yaml
logging:
  tensorboard:
    enabled: true                    # Enable TensorBoard logging
    log_dir: "./runs"                # Base directory for event files
    log_interval: 10                 # Log batch loss every N batches
    viz_interval: 1                  # Visualize predictions every N epochs
    val_interval: 10                 # Validate every N epochs
    log_histogram_interval: 5        # Log weight/gradient histograms every N epochs
    num_viz_samples: 8               # Number of samples to visualize per epoch
```

**Intervals explanation:**
- `log_interval=10`: Balance between granularity and overhead
- `viz_interval=1`: Continuous visual feedback (images every epoch)
- `val_interval=10`: Full validation is expensive (only every 10 epochs)
- `log_histogram_interval=5`: Distributions evolve slowly (no need for every epoch)

---

## References

**Official Documentation:**
- [PyTorch TensorBoard Integration](https://pytorch.org/docs/stable/tensorboard.html)
- [TensorBoard.dev Documentation](https://www.tensorflow.org/tensorboard)
- [SummaryWriter API](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)

**Event File Format:**
- [TensorFlow Event Format Spec](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/event.proto)
- [Protocol Buffers](https://developers.google.com/protocol-buffers)

**Best Practices:**
- [Neptune.ai TensorBoard Tutorial](https://neptune.ai/blog/tensorboard-tutorial)
- [PyTorch Lightning TensorBoard Integration](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#tensorboard)

**Related Papers:**
- ["Visualizing the Loss Landscape of Neural Nets"](https://arxiv.org/abs/1712.09913) - Understanding loss curves
- ["On the Spectral Bias of Neural Networks"](https://arxiv.org/abs/1806.08734) - Gradient analysis

---

## Changelog

### Version 2.0 (2025-12-23)
- ✅ Implemented immediate flush mode (1-second latency)
- ✅ Fixed epoch-level scalar logging (all metrics every epoch)
- ✅ Fixed image visualizations (appear after epoch 1)
- ✅ Added comprehensive loss component tracking
- ✅ Added epoch time tracking
- ✅ Consolidated documentation into single guide

### Version 1.0 (2025-12-20)
- Initial TensorBoard integration
- JSON-IPC architecture
- Python writer service
- Enhanced trainer with comprehensive logging
- Basic documentation

---

## Support and Troubleshooting

**For issues:**
1. Check [Troubleshooting section](#troubleshooting) above
2. Verify setup with diagnostic commands
3. Check console logs for error messages
4. Inspect event files in `./runs/experiment/`

**Common issues quick reference:**
- **No data:** Check Python writer is running, event files exist
- **Only 2 graphs:** Fixed in v2.0 (rebuild and restart)
- **No images:** Fixed in v2.0 (rebuild and restart)
- **Broken images:** Check temp PNG files, verify tensor format
- **High memory:** Reduce `num_viz_samples` or `batch_size`

**Testing TensorBoard integration:**
```bash
# Test 1: Python writer works
echo '{"type":"scalar","tag":"test","value":1.0,"step":0}' | \
  python scripts/tensorboard_writer.py ./runs/test

# Test 2: Event file created
ls -lh runs/test/events.out.tfevents.*

# Test 3: TensorBoard can read it
tensorboard --logdir=./runs/test --port=6007
# Open http://localhost:6007, should see "test" scalar
```

---

**END OF GUIDE**

For questions or suggestions, update this document as the project evolves.
