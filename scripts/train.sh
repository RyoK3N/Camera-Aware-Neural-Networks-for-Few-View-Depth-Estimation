#!/bin/bash
#
# Training Script for Camera-Aware Depth Estimation
#
# Usage:
#   ./scripts/train.sh baseline_unet
#   ./scripts/train.sh geometry_aware_full --gpu 0
#   ./scripts/train.sh intrinsics_only --debug
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
EXPERIMENT="baseline_unet"
CONFIG="configs/train_config.yaml"
GPU=0
DEBUG=false
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [EXPERIMENT] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  EXPERIMENT              Experiment name from config (default: baseline_unet)"
            echo ""
            echo "Options:"
            echo "  --config PATH          Config file path (default: configs/train_config.yaml)"
            echo "  --gpu ID               GPU ID to use (default: 0)"
            echo "  --debug                Enable debug mode (reduced dataset)"
            echo "  --resume PATH          Resume from checkpoint"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Available experiments:"
            echo "  baseline_unet"
            echo "  baseline_small"
            echo "  baseline_large"
            echo "  intrinsics_only"
            echo "  geometry_aware_full"
            echo "  geometry_aware_lightweight"
            echo "  ablation_rays_only"
            echo "  ablation_film_only"
            echo "  ablation_attention_only"
            exit 0
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Camera-Aware Depth Estimation - Training Launcher       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found${NC}"
    echo "Please run: mkdir build && cd build && cmake .. && make"
    exit 1
fi

# Check if training executable exists
if [ ! -f "build/train" ]; then
    echo -e "${RED}Error: training executable not found${NC}"
    echo "Please run: cd build && make train"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: config file not found: $CONFIG${NC}"
    exit 1
fi

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Experiment: $EXPERIMENT"
echo "  Config:     $CONFIG"
echo "  GPU:        $GPU"
echo "  Debug:      $DEBUG"
if [ -n "$RESUME" ]; then
    echo "  Resume:     $RESUME"
fi
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n 1
    echo ""
else
    echo -e "${YELLOW}Warning: nvidia-smi not found, using CPU${NC}"
    echo ""
fi

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Build command
CMD="./build/train --config $CONFIG --experiment $EXPERIMENT --gpu $GPU"

if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Print command
echo -e "${BLUE}Executing:${NC}"
echo "  $CMD"
echo ""

# Confirm execution
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo "════════════════════════════════════════════════════════════"
echo ""

# Run training
$CMD

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo ""
    echo "Results:"
    echo "  Checkpoints: checkpoints/$EXPERIMENT/"
    echo "  Logs:        logs/$EXPERIMENT/"
    echo ""
    echo "To evaluate the model, run:"
    echo "  ./scripts/evaluate.sh checkpoints/$EXPERIMENT/best_model.pt"
else
    echo -e "${RED}✗ Training failed with exit code $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
