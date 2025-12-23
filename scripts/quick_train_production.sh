#!/bin/bash
# Quick training script for Production GPU
# Launches training with TensorBoard monitoring

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Colors
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN}🚀 Production Training - Quick Start${NC}"
echo ""
echo -e "${GREEN}Config: configs/train_config_production.yaml${NC}"
echo -e "${GREEN}Device: CUDA (NVIDIA GPU)${NC}"
echo -e "${GREEN}Batch Size: 32${NC}"
echo -e "${GREEN}Image Size: 480×640${NC}"
echo ""

# Use the orchestration script
./scripts/train_with_monitoring.sh configs/train_config_production.yaml 6006
