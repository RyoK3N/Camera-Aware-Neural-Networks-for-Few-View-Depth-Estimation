#!/bin/bash
# Standalone TensorBoard launcher
# Opens TensorBoard in browser automatically

PORT="${1:-6006}"
LOGDIR="${2:-./runs}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN}📊 Starting TensorBoard${NC}"
echo ""
echo -e "${GREEN}Port: ${PORT}${NC}"
echo -e "${GREEN}Log Directory: ${LOGDIR}${NC}"
echo -e "${GREEN}URL: http://localhost:${PORT}${NC}"
echo ""

# Check if Python script exists
if [ -f "scripts/launch_tensorboard.py" ]; then
    python scripts/launch_tensorboard.py --port "${PORT}" --logdir "${LOGDIR}"
else
    # Fallback to direct tensorboard command
    echo "ℹ️  Using tensorboard directly..."
    tensorboard --logdir="${LOGDIR}" --port="${PORT}" --reload_interval=5
fi
