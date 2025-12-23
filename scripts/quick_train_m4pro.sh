#!/bin/bash
# Quick training script for Mac M4 Pro
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
echo -e "${BOLD}${CYAN}🍎 Mac M4 Pro Training - Quick Start${NC}"
echo ""
echo -e "${GREEN}Config: configs/train_config_m4pro.yaml${NC}"
echo -e "${GREEN}Device: MPS (Metal Performance Shaders)${NC}"
echo -e "${GREEN}Batch Size: 16${NC}"
echo -e "${GREEN}Image Size: 240×320${NC}"
echo ""

# Use the orchestration script
./scripts/train_with_monitoring.sh configs/train_config_m4pro.yaml 6006
