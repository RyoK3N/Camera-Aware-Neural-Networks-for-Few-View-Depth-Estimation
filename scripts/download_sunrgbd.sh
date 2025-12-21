#!/bin/bash

# SUN RGB-D Dataset Download Script
# Dataset: 10,335 RGB-D images from NYU Depth v2, Berkeley B3DO, and SUN3D
# Official Site: https://rgbd.cs.princeton.edu/

set -e

# Configuration
DATA_DIR="./data/sunrgbd"
DOWNLOAD_DIR="./data/downloads"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== SUN RGB-D Dataset Downloader ===${NC}"
echo ""
echo "Dataset Information:"
echo "  - 10,335 RGB-D images"
echo "  - Sources: NYU Depth v2, Berkeley B3DO, SUN3D"
echo "  - 4 different sensors"
echo "  - Indoor scene understanding"
echo ""

# Create directories
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p "$DATA_DIR"
mkdir -p "$DOWNLOAD_DIR"

echo ""
echo -e "${YELLOW}=== Download Options ===${NC}"
echo ""
echo "Choose your download method:"
echo ""
echo "1. Official SUN RGB-D Website (Recommended)"
echo "   - Complete dataset with toolbox"
echo "   - URL: https://rgbd.cs.princeton.edu/"
echo ""
echo "2. Reorganized Version (Easier to use)"
echo "   - Pre-processed structure"
echo "   - GitHub: https://github.com/chrischoy/SUN_RGBD"
echo ""
echo "3. Manual Setup"
echo "   - Create directory structure for manual download"
echo ""

read -p "Enter your choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}=== Official SUN RGB-D Download ===${NC}"
        echo ""
        echo "Please visit: https://rgbd.cs.princeton.edu/"
        echo ""
        echo "Download the following files:"
        echo "  1. SUNRGBD V1 (10,335 images) - Main dataset"
        echo "  2. SUNRGBDtoolbox - MATLAB toolbox"
        echo "  3. SUNRGBDMeta2DBB_v2.mat - 2D bounding boxes"
        echo "  4. SUNRGBDMeta3DBB_v2.mat - 3D bounding boxes"
        echo ""
        echo "After downloading, extract to: $DATA_DIR"
        echo ""

        # Create expected directory structure
        mkdir -p "$DATA_DIR/SUNRGBD"
        mkdir -p "$DATA_DIR/SUNRGBDtoolbox"

        echo -e "${YELLOW}Expected structure after extraction:${NC}"
        cat << 'EOF'
data/sunrgbd/
├── SUNRGBD/
│   ├── kv1/          # Kinect v1 data
│   ├── kv2/          # Kinect v2 data
│   ├── realsense/    # Intel RealSense data
│   └── xtion/        # Asus Xtion data
├── SUNRGBDtoolbox/   # MATLAB toolbox
├── SUNRGBDMeta2DBB_v2.mat
└── SUNRGBDMeta3DBB_v2.mat
EOF
        ;;

    2)
        echo ""
        echo -e "${GREEN}=== Reorganized SUN RGB-D Download ===${NC}"
        echo ""

        if ! command -v git &> /dev/null; then
            echo -e "${RED}Error: git is not installed${NC}"
            exit 1
        fi

        cd "$DOWNLOAD_DIR"

        echo "Cloning repository..."
        if [ -d "SUN_RGBD" ]; then
            echo "Repository already exists, pulling latest..."
            cd SUN_RGBD
            git pull
        else
            git clone https://github.com/chrischoy/SUN_RGBD.git
            cd SUN_RGBD
        fi

        echo ""
        echo "Making download script executable..."
        chmod +x download_and_extract.sh

        echo ""
        echo -e "${YELLOW}Running download script...${NC}"
        echo "This will download and organize the dataset."
        echo ""

        read -p "Continue with download? (y/n): " confirm
        if [ "$confirm" == "y" ]; then
            ./download_and_extract.sh

            echo ""
            echo "Moving data to $DATA_DIR..."
            cp -r SUNRGBD/* "$DATA_DIR/"

            echo -e "${GREEN}Download complete!${NC}"
        else
            echo "Download cancelled. You can run the script manually:"
            echo "  cd $DOWNLOAD_DIR/SUN_RGBD"
            echo "  ./download_and_extract.sh"
        fi
        ;;

    3)
        echo ""
        echo -e "${GREEN}=== Manual Setup ===${NC}"
        echo ""

        # Create comprehensive directory structure
        echo "Creating directory structure for manual download..."

        mkdir -p "$DATA_DIR/SUNRGBD/kv1"
        mkdir -p "$DATA_DIR/SUNRGBD/kv2"
        mkdir -p "$DATA_DIR/SUNRGBD/realsense"
        mkdir -p "$DATA_DIR/SUNRGBD/xtion"
        mkdir -p "$DATA_DIR/SUNRGBDtoolbox"
        mkdir -p "$DATA_DIR/processed/train/image"
        mkdir -p "$DATA_DIR/processed/train/depth"
        mkdir -p "$DATA_DIR/processed/train/intrinsics"
        mkdir -p "$DATA_DIR/processed/train/extrinsics"
        mkdir -p "$DATA_DIR/processed/val/image"
        mkdir -p "$DATA_DIR/processed/val/depth"
        mkdir -p "$DATA_DIR/processed/val/intrinsics"
        mkdir -p "$DATA_DIR/processed/val/extrinsics"
        mkdir -p "$DATA_DIR/processed/test/image"
        mkdir -p "$DATA_DIR/processed/test/depth"
        mkdir -p "$DATA_DIR/processed/test/intrinsics"
        mkdir -p "$DATA_DIR/processed/test/extrinsics"

        echo -e "${GREEN}Directory structure created!${NC}"
        echo ""
        echo -e "${YELLOW}Manual Download Instructions:${NC}"
        echo ""
        echo "1. Visit: https://rgbd.cs.princeton.edu/"
        echo ""
        echo "2. Download SUNRGBD V1 (main dataset)"
        echo "   - Contains 10,335 RGB-D images"
        echo "   - Extract to: $DATA_DIR/SUNRGBD/"
        echo ""
        echo "3. Download SUNRGBDtoolbox"
        echo "   - Contains MATLAB code and annotations"
        echo "   - Extract to: $DATA_DIR/SUNRGBDtoolbox/"
        echo ""
        echo "4. Expected data format per image:"
        echo "   - image/: RGB images (.jpg)"
        echo "   - depth/: Depth maps (.png)"
        echo "   - intrinsics.txt: Camera intrinsic matrix (3x3)"
        echo "   - extrinsics/: Camera extrinsic matrices"
        echo "   - scene.txt: Scene type information"
        echo ""
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Create README in data directory
cat > "$DATA_DIR/README.md" << 'EOF'
# SUN RGB-D Dataset

This directory contains the SUN RGB-D dataset for Camera-Aware Neural Networks project.

## Dataset Information

- **Total Images**: 10,335 RGB-D pairs
- **Sources**:
  - NYU Depth v2
  - Berkeley B3DO
  - SUN3D
- **Sensors**: 4 different types
  - Kinect v1
  - Kinect v2
  - Intel RealSense
  - Asus Xtion

## Directory Structure

### Original Format
```
SUNRGBD/
├── kv1/          # Kinect v1 data
├── kv2/          # Kinect v2 data
├── realsense/    # Intel RealSense data
└── xtion/        # Asus Xtion data
```

Each image directory contains:
- `image/` - RGB image (.jpg)
- `depth/` - Depth map (.png)
- `intrinsics.txt` - Camera intrinsic matrix (3x3)
- `extrinsics/` - Camera extrinsic parameters
- `scene.txt` - Scene type

### Processed Format
```
processed/
├── train/
│   ├── image/
│   ├── depth/
│   ├── intrinsics/
│   ├── extrinsics/
│   └── rays/
├── val/
│   └── ...
└── test/
    └── ...
```

## Camera Intrinsics Format

Intrinsics are stored in `intrinsics.txt` as a 3x3 matrix:
```
fx  0   cx
0   fy  cy
0   0   1
```

## Citations

If you use this dataset, please cite:

```bibtex
@inproceedings{song2015sun,
  title={SUN RGB-D: A RGB-D scene understanding benchmark suite},
  author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
  booktitle={CVPR},
  year={2015}
}

@inproceedings{silberman2012indoor,
  title={Indoor segmentation and support inference from rgbd images},
  author={Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob},
  booktitle={ECCV},
  year={2012}
}

@inproceedings{janoch2011category,
  title={A category-level 3-d object dataset: Putting the kinect to work},
  author={Janoch, Allison and Karayev, Sergey and Jia, Yangqing and Barron, Jonathan T and Fritz, Mario and Saenko, Kate and Darrell, Trevor},
  booktitle={ICCV Workshop},
  year={2011}
}

@inproceedings{xiao2013sun3d,
  title={SUN3D: A database of big spaces reconstructed using SfM and object labels},
  author={Xiao, Jianxiong and Owens, Andrew and Torralba, Antonio},
  booktitle={ICCV},
  year={2013}
}
```

## Data Processing Status

- [ ] Downloaded raw data
- [ ] Validated data integrity
- [ ] Precomputed ray directions
- [ ] Created train/val/test splits
- [ ] Generated data manifest

## Official Links

- Website: https://rgbd.cs.princeton.edu/
- Paper: SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite (CVPR 2015)
EOF

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Ensure data is downloaded to: $DATA_DIR"
echo "  2. Run validation: ./build/validate_data --data_dir $DATA_DIR"
echo "  3. Precompute ray directions: ./build/preprocess_rays --data_dir $DATA_DIR"
echo ""
echo "For more information, see: $DATA_DIR/README.md"
echo ""

# Create a simple Python script to help with data organization
cat > "$DATA_DIR/organize_data.py" << 'EOF'
#!/usr/bin/env python3
"""
SUN RGB-D Data Organization Script

This script helps organize the SUN RGB-D dataset into a standardized format
for the Camera-Aware Neural Networks project.

Usage:
    python organize_data.py --input ./SUNRGBD --output ./processed
"""

import os
import shutil
import argparse
from pathlib import Path

def organize_sunrgbd(input_dir, output_dir):
    """
    Organize SUN RGB-D data from original format to processed format.

    Args:
        input_dir: Path to original SUNRGBD directory
        output_dir: Path to output processed directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        for subdir in ['image', 'depth', 'intrinsics', 'extrinsics', 'rays']:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)

    print(f"Organizing data from {input_dir} to {output_dir}")
    print("This script is a template - please customize based on your needs")

    # TODO: Implement actual organization logic based on sensor types
    # This is a placeholder for the actual implementation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize SUN RGB-D dataset')
    parser.add_argument('--input', default='./SUNRGBD', help='Input directory')
    parser.add_argument('--output', default='./processed', help='Output directory')
    args = parser.parse_args()

    organize_sunrgbd(args.input, args.output)
EOF

chmod +x "$DATA_DIR/organize_data.py"

echo -e "${BLUE}Created helper script: $DATA_DIR/organize_data.py${NC}"
echo ""
