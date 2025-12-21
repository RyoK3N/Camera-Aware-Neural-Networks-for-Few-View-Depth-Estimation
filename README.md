# Camera-Aware Neural Networks for Few-View Depth Estimation

This project implements and validates camera-aware neural networks with geometric priors for few-view depth estimation using the **SUN RGB-D dataset**. The goal is to demonstrate that explicit geometric priors outperform naive conditioning and learned pose methods.

## Project Overview

**Research Question:** Can camera-aware neural networks with geometric priors significantly outperform naive conditioning and learned pose methods in few-view depth estimation?

**Hypothesis:** By incorporating camera intrinsics and ray directions as explicit geometric priors, neural networks can achieve better depth estimation performance with fewer input views.

**Dataset:** SUN RGB-D V1 (10,335 RGB-D images from NYU Depth v2, Berkeley B3DO, and SUN3D)

## Features

- **Ray Direction Computation**: Efficient C++ implementation using Eigen for computing per-pixel ray directions
- **ScanNet Data Loader**: Complete data loading pipeline with augmentation support
- **Geometric Priors**: Integration of camera intrinsics, extrinsics, and ray directions
- **Multiple Architectures**: Baseline U-Net, intrinsics-conditioned, and full geometry-aware networks

## Project Structure

```
Camera Matrix/
├── documents/              # Project documentation
│   ├── sprint_plan.md      # 4-week sprint plan
│   └── algorithms_and_theory.md  # Mathematical formulations
├── src/                    # Source code
│   ├── preprocessing/      # Ray direction computation
│   ├── data/              # Data loading and augmentation
│   ├── models/            # Neural network architectures
│   ├── layers/            # Custom layers (PCL, FiLM)
│   ├── loss/              # Loss functions
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Evaluation metrics
│   └── visualization/     # Visualization tools
├── scripts/               # Utility scripts
│   ├── download_scannet.sh    # Dataset download helper
│   └── validate_data.cpp      # Data validation tool
├── configs/               # Configuration files
│   └── data_config.yaml   # Data loading configuration
├── data/                  # Dataset directory
│   ├── scannet/           # ScanNet dataset
│   └── manifest/          # Data manifests
├── tests/                 # Unit tests
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── results/               # Experimental results
└── CMakeLists.txt         # Build configuration
```

## Dependencies

### Core Dependencies
- **C++ Compiler**: GCC 9+ or Clang 11+ with C++17 support
- **CMake**: 3.14 or higher
- **PyTorch C++ (LibTorch)**: 2.0 or higher
- **Eigen**: 3.4 or higher
- **OpenCV**: 4.5 or higher
- **nlohmann/json**: 3.2.0 or higher

### Python Dependencies (for visualization and analysis)
```bash
pip install matplotlib seaborn tensorboard pandas scipy
```

## Installation

### 1. Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake git
sudo apt-get install libeigen3-dev libopencv-dev
```

**macOS:**
```bash
brew install cmake eigen opencv
```

### 2. Install LibTorch

Download LibTorch from [PyTorch website](https://pytorch.org/):
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
export CMAKE_PREFIX_PATH=/path/to/libtorch
```

### 3. Build Project

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j$(nproc)
```

## Getting Started

### Week 1: Data Pipeline & Preprocessing (COMPLETED ✓)

#### 1. Download SUN RGB-D Dataset

**No approval required!** SUN RGB-D is freely available for download.

**Option 1: Official Website (Recommended)**
1. Visit [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/)
2. Download SUNRGBD V1 (10,335 images)
3. Download SUNRGBDtoolbox
4. Extract to `./data/sunrgbd/`

**Option 2: Automated Script**
```bash
bash scripts/download_sunrgbd.sh
```

The script provides three download methods:
- Official website download (with instructions)
- Reorganized version from GitHub (automated)
- Manual setup (creates directory structure)

#### 2. Validate Dataset

```bash
./build/validate_sunrgbd ./data/sunrgbd ./data/sunrgbd_manifest.json
```

This will:
- Validate all 10,335 RGB-D images
- Check file completeness across 4 sensor types
- Verify RGB-depth-intrinsics alignment
- Validate intrinsic and extrinsic matrices
- Generate a comprehensive data manifest JSON file

#### 3. Precompute Ray Directions

```bash
./build/preprocess_rays --data_dir ./data/sunrgbd
```

## Usage Examples

### Data Loading

```cpp
#include "sunrgbd_loader.h"

using namespace camera_aware_depth;

// Create data loader
auto loader = std::make_shared<SunRGBDLoader>(
    "./data/sunrgbd",                    // data directory
    "./data/sunrgbd_manifest.json",      // manifest path
    "train"                              // split (train or test)
);

// Filter by sensor type (optional)
loader->filterBySensorType({"kv1", "kv2"});  // Use only Kinect v1 and v2

// Enable augmentation
AugmentationConfig aug_config;
aug_config.enable_random_crop = true;
aug_config.enable_horizontal_flip = true;
aug_config.enable_color_jitter = true;
loader->enableAugmentation(aug_config);

// Set target dimensions
loader->setTargetDimensions(240, 320);

// Load a sample
SunRGBDSample sample = loader->getSample(0);
std::cout << "RGB shape: " << sample.rgb.sizes() << std::endl;
std::cout << "Depth shape: " << sample.depth.sizes() << std::endl;
std::cout << "Sensor type: " << sample.sensor_type << std::endl;
std::cout << "Scene type: " << sample.scene_type << std::endl;
```

### Ray Direction Computation

```cpp
#include "ray_direction_computer.h"

using namespace camera_aware_depth;

// Load camera intrinsics
Eigen::Matrix3f K = RayDirectionComputer::loadIntrinsics("intrinsic.txt");

// Compute ray directions
RayDirectionComputer computer;
Eigen::MatrixXf rays = computer.computeRayDirections(K, 480, 640);

// Save to file
computer.saveRayDirections(rays, 480, 640, "rays.bin");
```

## Configuration

Edit `configs/data_config.yaml` to customize:
- Target image dimensions
- Data augmentation parameters
- Batch size and dataloader settings
- File formats and paths

## Testing

Run unit tests:
```bash
./build/test_ray_directions
./build/test_data_loader
```

## Results

Results will be saved in the `results/` directory:
- `results/quantitative_results.csv`: Numerical metrics
- `results/visualizations/`: Visual comparisons
- `results/tables/`: LaTeX tables for paper

## Project Timeline

| Week | Focus Area | Status |
|------|------------|--------|
| 1 | Data Pipeline & Preprocessing | ✓ Completed |
| 2 | Baseline & Model Implementation | In Progress |
| 3 | Training & Ablation Studies | Pending |
| 4 | Evaluation & Analysis | Pending |

**Total Estimated Hours:** 120-150 hours
**Sprint Duration:** 4 weeks

## Documentation

- **Sprint Plan**: See `documents/sprint_plan.md` for detailed weekly tasks
- **Algorithms & Theory**: See `documents/algorithms_and_theory.md` for mathematical formulations

## Contributing

This is a research project. For questions or suggestions, please open an issue.

## Dataset Information

### SUN RGB-D V1

- **Total Images**: 10,335 RGB-D pairs
- **Sources**:
  - NYU Depth v2 (indoor scenes)
  - Berkeley B3DO (indoor scenes)
  - SUN3D (indoor scenes)
- **Sensors**: 4 types
  - Kinect v1 (3,389 images)
  - Kinect v2 (3,389 images)
  - Intel RealSense (1,574 images)
  - Asus Xtion (1,983 images)
- **Resolution**: Varies by sensor (typically 640×480)
- **Official Split**: 5,285 train / 5,050 test
- **License**: Free for research use

### Citations

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

## References

- **SUN RGB-D Dataset**: [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/)
- **PyTorch C++ Frontend**: [https://pytorch.org/cppdocs/](https://pytorch.org/cppdocs/)
- **Eigen Library**: [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)

## License

This project is for research purposes only.

## Acknowledgments

- ScanNet dataset creators for providing the dataset
- PyTorch team for LibTorch C++ frontend
- Eigen and OpenCV communities

---

**Status:** Week 1 Completed ✓

**Next Steps:**
1. Begin implementing baseline U-Net architecture
2. Implement custom layers (PCL, FiLM)
3. Set up training pipeline

For detailed information, refer to the sprint plan in `documents/sprint_plan.md`.
