#include <iostream>
#include <memory>
#include "../src/data/sunrgbd_loader.h"

using namespace camera_aware_depth;

int main() {
    try {
        std::cout << "=== Testing SUN RGB-D Data Loader ===" << std::endl;

        // Create loader
        std::string data_dir = "./data/sunrgbd";
        std::string manifest_path = "./data/sunrgbd_manifest.json";
        std::string split = "train";

        std::cout << "Creating data loader..." << std::endl;
        std::cout << "  Data dir: " << data_dir << std::endl;
        std::cout << "  Manifest: " << manifest_path << std::endl;
        std::cout << "  Split: " << split << std::endl;

        auto loader = std::make_shared<SunRGBDLoader>(data_dir, manifest_path, split);

        std::cout << "\n" << loader->getStatistics() << std::endl;

        // Test loading a single sample
        if (loader->size() > 0) {
            std::cout << "\nTesting sample loading..." << std::endl;
            std::cout << "Loading sample 0..." << std::endl;

            auto sample = loader->getSample(0);

            std::cout << "✓ Sample loaded successfully!" << std::endl;
            std::cout << "  Image path: " << sample.image_path << std::endl;
            std::cout << "  Sensor type: " << sample.sensor_type << std::endl;
            std::cout << "  RGB shape: [" << sample.rgb.size(0) << ", "
                      << sample.rgb.size(1) << ", " << sample.rgb.size(2) << "]" << std::endl;
            std::cout << "  Depth shape: [" << sample.depth.size(0) << ", "
                      << sample.depth.size(1) << ", " << sample.depth.size(2) << "]" << std::endl;
            std::cout << "  Intrinsics shape: [" << sample.intrinsics.size(0) << ", "
                      << sample.intrinsics.size(1) << "]" << std::endl;

            if (!sample.scene_type.empty()) {
                std::cout << "  Scene type: " << sample.scene_type << std::endl;
            }

            std::cout << "\n=== Test PASSED ===" << std::endl;
        } else {
            std::cout << "ERROR: No samples found in loader!" << std::endl;
            return 1;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        std::cerr << "=== Test FAILED ===" << std::endl;
        return 1;
    }
}
