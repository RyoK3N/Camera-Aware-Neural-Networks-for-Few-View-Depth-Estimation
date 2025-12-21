#!/bin/bash

###############################################################################
# Environment Setup Script
#
# Checks for required dependencies and provides installation instructions
# for the Camera-Aware Depth Estimation project
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==================================================================="
echo -e "    Camera-Aware Depth Estimation - Environment Setup"
echo -e "===================================================================${NC}\n"

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo -e "${GREEN}Detected OS: macOS${NC}"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "${GREEN}Detected OS: Linux${NC}"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo ""

###############################################################################
# Check Dependencies
###############################################################################

echo -e "${BLUE}Checking dependencies...${NC}\n"

MISSING_DEPS=()
WARNINGS=()

# 1. Check CMake
echo -n "Checking CMake (>= 3.14)... "
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d'.' -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d'.' -f2)

    if [ "$CMAKE_MAJOR" -gt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -ge 14 ]); then
        echo -e "${GREEN}✓ Found CMake $CMAKE_VERSION${NC}"
    else
        echo -e "${RED}✗ CMake version too old ($CMAKE_VERSION < 3.14)${NC}"
        MISSING_DEPS+=("cmake")
    fi
else
    echo -e "${RED}✗ Not found${NC}"
    MISSING_DEPS+=("cmake")
fi

# 2. Check C++ Compiler
echo -n "Checking C++ compiler (C++17)... "
if command -v clang++ &> /dev/null; then
    COMPILER_VERSION=$(clang++ --version | head -n1)
    echo -e "${GREEN}✓ Found clang++${NC}"
    echo "  Version: $COMPILER_VERSION"
elif command -v g++ &> /dev/null; then
    COMPILER_VERSION=$(g++ --version | head -n1)
    echo -e "${GREEN}✓ Found g++${NC}"
    echo "  Version: $COMPILER_VERSION"
else
    echo -e "${RED}✗ Not found${NC}"
    MISSING_DEPS+=("compiler")
fi

# 3. Check Python
echo -n "Checking Python 3... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓ Found Python $PYTHON_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Not found (optional for helper scripts)${NC}"
    WARNINGS+=("python3")
fi

# 4. Check Homebrew (macOS) or apt (Linux)
if [ "$OS" == "macos" ]; then
    echo -n "Checking Homebrew... "
    if command -v brew &> /dev/null; then
        BREW_VERSION=$(brew --version | head -n1)
        echo -e "${GREEN}✓ Found Homebrew${NC}"
    else
        echo -e "${YELLOW}⚠ Not found (recommended for dependency management)${NC}"
        WARNINGS+=("homebrew")
    fi
fi

# 5. Check LibTorch
echo -n "Checking LibTorch... "
LIBTORCH_FOUND=false

# Common LibTorch installation paths
LIBTORCH_PATHS=(
    "/usr/local/libtorch"
    "/opt/libtorch"
    "$HOME/libtorch"
    "$HOME/.local/libtorch"
    "/usr/local/Cellar/pytorch"
)

for path in "${LIBTORCH_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/share/cmake/Torch/TorchConfig.cmake" ]; then
        echo -e "${GREEN}✓ Found at $path${NC}"
        LIBTORCH_FOUND=true
        export Torch_DIR="$path/share/cmake/Torch"
        break
    fi
done

if [ "$LIBTORCH_FOUND" = false ]; then
    echo -e "${RED}✗ Not found${NC}"
    MISSING_DEPS+=("libtorch")
fi

# 6. Check Eigen3
echo -n "Checking Eigen3 (>= 3.4)... "
EIGEN_FOUND=false

if [ "$OS" == "macos" ]; then
    if [ -d "/usr/local/include/eigen3" ] || [ -d "/opt/homebrew/include/eigen3" ]; then
        echo -e "${GREEN}✓ Found${NC}"
        EIGEN_FOUND=true
    fi
elif [ "$OS" == "linux" ]; then
    if [ -d "/usr/include/eigen3" ] || [ -d "/usr/local/include/eigen3" ]; then
        echo -e "${GREEN}✓ Found${NC}"
        EIGEN_FOUND=true
    fi
fi

if [ "$EIGEN_FOUND" = false ]; then
    echo -e "${RED}✗ Not found${NC}"
    MISSING_DEPS+=("eigen3")
fi

# 7. Check OpenCV
echo -n "Checking OpenCV (>= 4.5)... "
if pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo -e "${GREEN}✓ Found OpenCV $OPENCV_VERSION${NC}"
elif command -v opencv_version &> /dev/null; then
    OPENCV_VERSION=$(opencv_version)
    echo -e "${GREEN}✓ Found OpenCV $OPENCV_VERSION${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    MISSING_DEPS+=("opencv")
fi

# 8. Check CUDA (optional)
echo -n "Checking CUDA (optional)... "
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    echo -e "${GREEN}✓ Found CUDA $CUDA_VERSION${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1
else
    echo -e "${YELLOW}⚠ Not found (will use CPU)${NC}"
    WARNINGS+=("cuda")
fi

echo ""

###############################################################################
# Installation Instructions
###############################################################################

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${RED}==================================================================="
    echo -e "                Missing Required Dependencies"
    echo -e "===================================================================${NC}\n"

    echo -e "${YELLOW}The following dependencies are missing:${NC}"
    for dep in "${MISSING_DEPS[@]}"; do
        echo -e "  - $dep"
    done
    echo ""

    echo -e "${BLUE}Installation Instructions:${NC}\n"

    if [ "$OS" == "macos" ]; then
        echo -e "${GREEN}macOS Installation Commands:${NC}\n"

        if [[ " ${MISSING_DEPS[@]} " =~ " homebrew " ]]; then
            echo "# Install Homebrew"
            echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " cmake " ]]; then
            echo "# Install CMake"
            echo "brew install cmake"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " compiler " ]]; then
            echo "# Install Command Line Tools (includes clang++)"
            echo "xcode-select --install"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " eigen3 " ]]; then
            echo "# Install Eigen3"
            echo "brew install eigen"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " opencv " ]]; then
            echo "# Install OpenCV"
            echo "brew install opencv"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " libtorch " ]]; then
            echo "# Install LibTorch (PyTorch C++)"
            echo "# Option 1: Download pre-built (CPU version)"
            echo "cd ~"
            echo "wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip"
            echo "unzip libtorch-macos-2.1.0.zip"
            echo "export Torch_DIR=~/libtorch/share/cmake/Torch"
            echo ""
            echo "# Option 2: Install via PyTorch package (if you have Python)"
            echo "pip3 install torch torchvision"
            echo "# Then link to LibTorch from Python installation"
            echo ""
        fi

    elif [ "$OS" == "linux" ]; then
        echo -e "${GREEN}Linux Installation Commands:${NC}\n"

        if [[ " ${MISSING_DEPS[@]} " =~ " cmake " ]]; then
            echo "# Install CMake"
            echo "sudo apt-get update"
            echo "sudo apt-get install cmake"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " compiler " ]]; then
            echo "# Install g++"
            echo "sudo apt-get install build-essential"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " eigen3 " ]]; then
            echo "# Install Eigen3"
            echo "sudo apt-get install libeigen3-dev"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " opencv " ]]; then
            echo "# Install OpenCV"
            echo "sudo apt-get install libopencv-dev python3-opencv"
            echo ""
        fi

        if [[ " ${MISSING_DEPS[@]} " =~ " libtorch " ]]; then
            echo "# Install LibTorch (PyTorch C++)"
            echo "# Download pre-built (CPU version)"
            echo "cd ~"
            echo "wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
            echo "unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip"
            echo "export Torch_DIR=~/libtorch/share/cmake/Torch"
            echo ""
            echo "# For CUDA version (if you have NVIDIA GPU):"
            echo "wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
            echo ""
        fi
    fi

    echo -e "${YELLOW}After installing dependencies, run this script again to verify.${NC}"
    echo ""
    exit 1
fi

###############################################################################
# Success Summary
###############################################################################

echo -e "${GREEN}==================================================================="
echo -e "                 All Required Dependencies Found!"
echo -e "===================================================================${NC}\n"

if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Optional components:${NC}"
    for warn in "${WARNINGS[@]}"; do
        case $warn in
            cuda)
                echo -e "  - CUDA not found: Will use CPU for training (slower)"
                ;;
            python3)
                echo -e "  - Python3 not found: Some helper scripts may not work"
                ;;
            homebrew)
                echo -e "  - Homebrew not found: Manual dependency management required"
                ;;
        esac
    done
    echo ""
fi

echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Build the project:"
echo "   mkdir -p build && cd build"
echo "   cmake .."
echo "   make -j\$(nproc)"
echo ""
echo "2. Validate the dataset:"
echo "   ./build/validate_sunrgbd ./data/sunrgbd"
echo ""
echo "3. Run tests:"
echo "   ./build/test_ray_directions"
echo "   ./build/test_data_loader"
echo "   ./build/test_models"
echo ""
echo "4. Preprocess ray directions:"
echo "   ./build/preprocess_rays --data_dir ./data/sunrgbd"
echo ""
echo "5. Start training:"
echo "   bash scripts/train.sh"
echo ""

echo -e "${GREEN}Environment setup complete!${NC}"
