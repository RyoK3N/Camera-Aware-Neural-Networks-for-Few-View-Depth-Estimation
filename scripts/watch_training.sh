#!/bin/bash
# Watch training logs in real-time
# Alternative to using the Python monitor

LOG_FILE="${1:-./logs/training.log}"
METRICS_FILE="./logs/metrics.csv"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN}👀 Watching Training Logs${NC}"
echo ""
echo -e "${GREEN}Log file: ${LOG_FILE}${NC}"
echo ""

# Check if Python monitor is available
if [ -f "scripts/monitor_training.py" ] && command -v python &> /dev/null; then
    echo "Using Python monitor (interactive)..."
    echo ""
    python scripts/monitor_training.py --log "${LOG_FILE}" --metrics "${METRICS_FILE}"
else
    # Fallback to tail -f
    echo "Using tail -f (basic)..."
    echo ""
    if [ -f "${LOG_FILE}" ]; then
        tail -f "${LOG_FILE}"
    else
        echo "⏳ Waiting for log file: ${LOG_FILE}"
        while [ ! -f "${LOG_FILE}" ]; do
            sleep 1
        done
        tail -f "${LOG_FILE}"
    fi
fi
