#!/bin/bash

###############################################################################
# Evaluation Launcher Script
#
# Interactive script for evaluating trained depth estimation models
#
# Features:
# - Checkpoint selection and validation
# - Configuration management
# - Visualization options
# - Batch evaluation support
# - Results organization
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/results"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"

echo -e "${BLUE}==================================================================="
echo -e "          Camera-Aware Depth Estimation - Evaluation"
echo -e "===================================================================${NC}\n"

###############################################################################
# Check Prerequisites
###############################################################################

echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found${NC}"
    echo "Please build the project first:"
    echo "  mkdir build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# Check if evaluate executable exists
if [ ! -f "$BUILD_DIR/evaluate" ]; then
    echo -e "${RED}Error: evaluate executable not found${NC}"
    echo "Please build the project first:"
    echo "  cd build && make -j\$(nproc)"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected, will use CPU${NC}"
fi

echo ""

###############################################################################
# Find Available Checkpoints
###############################################################################

echo -e "${YELLOW}Scanning for trained models...${NC}"

# Find all .pt checkpoint files
mapfile -t CHECKPOINT_FILES < <(find "$CHECKPOINTS_DIR" -name "*.pt" 2>/dev/null | sort)

if [ ${#CHECKPOINT_FILES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No checkpoint files found in $CHECKPOINTS_DIR${NC}"
    echo "Please train a model first using scripts/train.sh"
    exit 1
fi

echo -e "${GREEN}Found ${#CHECKPOINT_FILES[@]} checkpoint(s)${NC}\n"

###############################################################################
# Checkpoint Selection
###############################################################################

echo -e "${BLUE}Available Checkpoints:${NC}"
for i in "${!CHECKPOINT_FILES[@]}"; do
    checkpoint="${CHECKPOINT_FILES[$i]}"
    basename=$(basename "$checkpoint")
    dirname=$(basename "$(dirname "$checkpoint")")

    # Get file size and modification time
    size=$(du -h "$checkpoint" | cut -f1)
    modtime=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$checkpoint" 2>/dev/null || stat -c "%y" "$checkpoint" 2>/dev/null | cut -d'.' -f1)

    echo -e "  ${GREEN}[$((i+1))]${NC} $dirname/$basename"
    echo -e "      Size: $size, Modified: $modtime"
done

echo ""
read -p "Select checkpoint number (or 'a' for all): " checkpoint_choice

if [ "$checkpoint_choice" == "a" ] || [ "$checkpoint_choice" == "A" ]; then
    EVAL_MODE="batch"
    SELECTED_CHECKPOINTS=("${CHECKPOINT_FILES[@]}")
    echo -e "${GREEN}Selected: All checkpoints (batch mode)${NC}"
else
    EVAL_MODE="single"
    idx=$((checkpoint_choice - 1))
    if [ $idx -lt 0 ] || [ $idx -ge ${#CHECKPOINT_FILES[@]} ]; then
        echo -e "${RED}Error: Invalid selection${NC}"
        exit 1
    fi
    SELECTED_CHECKPOINTS=("${CHECKPOINT_FILES[$idx]}")
    echo -e "${GREEN}Selected: $(basename "${CHECKPOINT_FILES[$idx]}")${NC}"
fi

echo ""

###############################################################################
# Configuration Selection
###############################################################################

echo -e "${YELLOW}Detecting configuration files...${NC}"

# Try to find corresponding config files
CONFIGS_DIR="$PROJECT_ROOT/configs"
CONFIG_FOUND=false

for checkpoint in "${SELECTED_CHECKPOINTS[@]}"; do
    # Try to infer config from checkpoint path
    exp_name=$(basename "$(dirname "$checkpoint")")

    # Look for matching config
    if [ -f "$CONFIGS_DIR/${exp_name}.yaml" ]; then
        CONFIG_FILE="$CONFIGS_DIR/${exp_name}.yaml"
        CONFIG_FOUND=true
        break
    fi
done

if [ "$CONFIG_FOUND" = false ]; then
    # List available configs
    echo -e "${BLUE}Available configurations:${NC}"
    mapfile -t CONFIG_FILES < <(find "$CONFIGS_DIR" -name "*.yaml" | sort)

    for i in "${!CONFIG_FILES[@]}"; do
        echo -e "  ${GREEN}[$((i+1))]${NC} $(basename "${CONFIG_FILES[$i]}")"
    done

    echo ""
    read -p "Select configuration number: " config_choice
    idx=$((config_choice - 1))

    if [ $idx -lt 0 ] || [ $idx -ge ${#CONFIG_FILES[@]} ]; then
        echo -e "${RED}Error: Invalid selection${NC}"
        exit 1
    fi

    CONFIG_FILE="${CONFIG_FILES[$idx]}"
fi

echo -e "${GREEN}Using config: $(basename "$CONFIG_FILE")${NC}"
echo ""

###############################################################################
# Evaluation Options
###############################################################################

echo -e "${YELLOW}Evaluation Options:${NC}"

# Number of visualizations
read -p "Number of visualizations to generate (default: 50): " num_vis
num_vis=${num_vis:-50}

# Colormap selection
echo ""
echo "Colormap options:"
echo "  1) viridis (default, perceptually uniform)"
echo "  2) plasma (perceptually uniform)"
echo "  3) magma (perceptually uniform)"
echo "  4) inferno (perceptually uniform)"
echo "  5) turbo (rainbow-like)"
read -p "Select colormap (1-5, default: 1): " colormap_choice
colormap_choice=${colormap_choice:-1}

case $colormap_choice in
    1) COLORMAP="viridis" ;;
    2) COLORMAP="plasma" ;;
    3) COLORMAP="magma" ;;
    4) COLORMAP="inferno" ;;
    5) COLORMAP="turbo" ;;
    *) COLORMAP="viridis" ;;
esac

echo -e "${GREEN}Selected colormap: $COLORMAP${NC}"
echo ""

# Save predictions option
read -p "Save all depth predictions? (y/N): " save_preds
SAVE_PREDS_FLAG=""
if [ "$save_preds" == "y" ] || [ "$save_preds" == "Y" ]; then
    SAVE_PREDS_FLAG="--save-predictions"
    echo -e "${GREEN}Will save all predictions${NC}"
fi

echo ""

###############################################################################
# Run Evaluation
###############################################################################

echo -e "${BLUE}==================================================================="
echo -e "                    Starting Evaluation"
echo -e "===================================================================${NC}\n"

# Function to evaluate a single checkpoint
evaluate_checkpoint() {
    local checkpoint=$1
    local output_dir=$2

    echo -e "${YELLOW}Evaluating: $(basename "$checkpoint")${NC}"
    echo "Output directory: $output_dir"
    echo ""

    # Run evaluation
    "$BUILD_DIR/evaluate" \
        --checkpoint "$checkpoint" \
        --config "$CONFIG_FILE" \
        --output "$output_dir" \
        --num-vis "$num_vis" \
        --colormap "$COLORMAP" \
        $SAVE_PREDS_FLAG

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed successfully${NC}"
        echo -e "Results saved to: $output_dir"
        return 0
    else
        echo -e "${RED}✗ Evaluation failed with exit code $exit_code${NC}"
        return $exit_code
    fi
}

# Execute evaluation
if [ "$EVAL_MODE" == "single" ]; then
    # Single checkpoint evaluation
    checkpoint="${SELECTED_CHECKPOINTS[0]}"
    exp_name=$(basename "$(dirname "$checkpoint")")
    ckpt_name=$(basename "$checkpoint" .pt)
    output_dir="$RESULTS_DIR/eval_${exp_name}_${ckpt_name}"

    evaluate_checkpoint "$checkpoint" "$output_dir"

    echo ""
    echo -e "${BLUE}==================================================================="
    echo -e "                   Evaluation Complete"
    echo -e "===================================================================${NC}\n"

    # Offer to open results
    if [ -f "$output_dir/evaluation_report.txt" ]; then
        read -p "View evaluation report? (Y/n): " view_report
        if [ "$view_report" != "n" ] && [ "$view_report" != "N" ]; then
            cat "$output_dir/evaluation_report.txt"
        fi
    fi

else
    # Batch evaluation
    echo -e "${YELLOW}Running batch evaluation on ${#SELECTED_CHECKPOINTS[@]} checkpoint(s)${NC}\n"

    SUCCESS_COUNT=0
    FAIL_COUNT=0

    for checkpoint in "${SELECTED_CHECKPOINTS[@]}"; do
        exp_name=$(basename "$(dirname "$checkpoint")")
        ckpt_name=$(basename "$checkpoint" .pt)
        output_dir="$RESULTS_DIR/eval_${exp_name}_${ckpt_name}"

        echo ""
        echo -e "${BLUE}--- Checkpoint $((SUCCESS_COUNT + FAIL_COUNT + 1))/${#SELECTED_CHECKPOINTS[@]} ---${NC}"

        if evaluate_checkpoint "$checkpoint" "$output_dir"; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
        fi

        echo ""
    done

    echo ""
    echo -e "${BLUE}==================================================================="
    echo -e "                Batch Evaluation Complete"
    echo -e "===================================================================${NC}\n"
    echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
    if [ $FAIL_COUNT -gt 0 ]; then
        echo -e "${RED}Failed: $FAIL_COUNT${NC}"
    fi
    echo ""
    echo "All results saved to: $RESULTS_DIR"
fi

echo ""
echo -e "${GREEN}Evaluation complete!${NC}"
echo ""

# Offer to run comparison if multiple results exist
EVAL_RESULTS=($(find "$RESULTS_DIR" -name "metrics.csv" -type f 2>/dev/null))
if [ ${#EVAL_RESULTS[@]} -gt 1 ]; then
    echo -e "${YELLOW}Found ${#EVAL_RESULTS[@]} evaluation results${NC}"
    read -p "Run comparison analysis? (Y/n): " run_comparison

    if [ "$run_comparison" != "n" ] && [ "$run_comparison" != "N" ]; then
        if [ -f "$SCRIPT_DIR/compare_models.sh" ]; then
            bash "$SCRIPT_DIR/compare_models.sh"
        else
            echo -e "${YELLOW}Comparison script not found${NC}"
        fi
    fi
fi

echo ""
echo -e "${BLUE}==================================================================${NC}"
