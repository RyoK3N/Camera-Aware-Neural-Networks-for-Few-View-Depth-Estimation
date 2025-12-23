#!/bin/bash
# Training Orchestration Script
# Launches training with TensorBoard and monitoring in separate terminals

set -e

# Configuration
CONFIG_FILE="${1:-configs/train_config_m4pro.yaml}"
TENSORBOARD_PORT="${2:-6006}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
LOGS_DIR="${PROJECT_DIR}/logs"
RUNS_DIR="${PROJECT_DIR}/runs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print with color
print_header() {
    echo -e "${BOLD}${BLUE}=================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}=================================${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running on macOS
is_macos() {
    [[ "$OSTYPE" == "darwin"* ]]
}

# Create directories
mkdir -p "${LOGS_DIR}"
mkdir -p "${RUNS_DIR}"

# Print header
clear
print_header "🚀 Camera-Aware Depth Estimation Training"
echo ""
print_info "Project Directory: ${PROJECT_DIR}"
print_info "Config File: ${CONFIG_FILE}"
print_info "TensorBoard Port: ${TENSORBOARD_PORT}"
echo ""

# Check if build exists
if [ ! -f "${BUILD_DIR}/train" ]; then
    print_error "Training executable not found!"
    print_info "Building project..."
    cd "${BUILD_DIR}"
    cmake .. && make train -j8
    cd "${PROJECT_DIR}"
    print_success "Build complete"
    echo ""
fi

# Check if config exists
if [ ! -f "${PROJECT_DIR}/${CONFIG_FILE}" ]; then
    print_error "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Check for Python and TensorBoard
if ! command -v python &> /dev/null; then
    print_warning "python not found. TensorBoard will not be launched automatically."
    SKIP_TENSORBOARD=1
else
    if ! python -c "import tensorboard" 2>/dev/null; then
        print_warning "TensorBoard not installed. Install with: conda install tensorboard"
        SKIP_TENSORBOARD=1
    fi
fi

# Function to open new terminal window
open_terminal() {
    local title="$1"
    local command="$2"

    if is_macos; then
        # macOS: Use osascript to open Terminal
        osascript <<EOF
tell application "Terminal"
    do script "cd '${PROJECT_DIR}' && ${command}"
    set custom title of front window to "${title}"
    activate
end tell
EOF
    else
        # Linux: Try common terminal emulators
        if command -v gnome-terminal &> /dev/null; then
            gnome-terminal --title="${title}" -- bash -c "cd '${PROJECT_DIR}' && ${command}; exec bash"
        elif command -v xterm &> /dev/null; then
            xterm -title "${title}" -e "cd '${PROJECT_DIR}' && ${command}; bash" &
        elif command -v konsole &> /dev/null; then
            konsole --title "${title}" -e bash -c "cd '${PROJECT_DIR}' && ${command}; bash" &
        else
            print_warning "Could not open terminal. Please run manually:"
            echo "  ${command}"
        fi
    fi
}

# Launch TensorBoard in new terminal
if [ -z "$SKIP_TENSORBOARD" ]; then
    print_info "Launching TensorBoard on port ${TENSORBOARD_PORT}..."

    # Detect conda environment and base path
    CONDA_ENV="${CONDA_DEFAULT_ENV:-synexian}"
    CONDA_BASE="/opt/homebrew/Caskroom/miniconda/base"

    open_terminal "TensorBoard - Port ${TENSORBOARD_PORT}" \
        "source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && python scripts/launch_tensorboard.py --port ${TENSORBOARD_PORT} --logdir ./runs"
    sleep 2
    print_success "TensorBoard launched"
else
    print_info "Skipping TensorBoard launch"
fi

# Ask user if they want live monitoring
echo ""
read -p "$(echo -e ${CYAN}Launch live log monitor? [Y/n]:${NC} )" -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    print_info "Launching log monitor..."
    open_terminal "Training Monitor" \
        "source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && python scripts/monitor_training.py --dashboard --refresh 5"
    sleep 1
    print_success "Monitor launched"
fi

# Summary
echo ""
print_header "📋 Training Setup Complete"
echo ""
print_success "TensorBoard: http://localhost:${TENSORBOARD_PORT}"
print_success "Log file: ${LOGS_DIR}/training.log"
print_success "Checkpoints: ${PROJECT_DIR}/checkpoints"
echo ""
print_info "Starting training in 3 seconds..."
echo ""
sleep 3

# Start training in current terminal
print_header "🎯 Starting Training"
echo ""
cd "${PROJECT_DIR}"

# Run training
"${BUILD_DIR}/train" \
    --config "${CONFIG_FILE}" \
    --tensorboard true

# Training completed
echo ""
print_header "🎉 Training Complete"
echo ""
print_success "Check results in:"
print_info "  • TensorBoard: http://localhost:${TENSORBOARD_PORT}"
print_info "  • Logs: ${LOGS_DIR}/"
print_info "  • Checkpoints: ${PROJECT_DIR}/checkpoints/"
echo ""
