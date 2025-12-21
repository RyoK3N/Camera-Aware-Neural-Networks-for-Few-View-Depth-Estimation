#include "ray_direction_computer.h"
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --data_dir <path_to_scannet_data>" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_dir <path>    Path to ScanNet data directory" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " --data_dir ./data/scannet" << std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data_dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (data_dir.empty()) {
        std::cerr << "Error: --data_dir is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if (!fs::exists(data_dir)) {
        std::cerr << "Error: Data directory does not exist: " << data_dir << std::endl;
        return 1;
    }

    std::cout << "=== Ray Direction Preprocessing Tool ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << std::endl;

    camera_aware_depth::RayDirectionComputer computer;

    // Iterate through all scenes
    int scenes_processed = 0;
    int total_frames = 0;

    for (const auto& scene_entry : fs::directory_iterator(data_dir)) {
        if (!scene_entry.is_directory()) continue;

        std::string scene_name = scene_entry.path().filename().string();
        if (scene_name.find("scene") != 0) continue;

        std::string scene_path = scene_entry.path().string();
        std::cout << "\nProcessing scene: " << scene_name << std::endl;

        // Load intrinsics
        std::string intrinsic_path = scene_path + "/intrinsic.txt";
        if (!fs::exists(intrinsic_path)) {
            std::cerr << "  Warning: intrinsic.txt not found, skipping scene" << std::endl;
            continue;
        }

        Eigen::Matrix3f K;
        try {
            K = camera_aware_depth::RayDirectionComputer::loadIntrinsics(intrinsic_path);
        } catch (const std::exception& e) {
            std::cerr << "  Error loading intrinsics: " << e.what() << std::endl;
            continue;
        }

        // Create rays directory
        std::string rays_dir = scene_path + "/rays";
        fs::create_directories(rays_dir);

        // Get image dimensions from a sample color image
        std::string color_dir = scene_path + "/color";
        if (!fs::exists(color_dir)) {
            std::cerr << "  Warning: color directory not found, skipping scene" << std::endl;
            continue;
        }

        // Find first image to get dimensions
        int height = 480;  // Default ScanNet resolution
        int width = 640;

        // Get all frame IDs
        std::vector<std::string> frame_ids;
        for (const auto& entry : fs::directory_iterator(color_dir)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                frame_ids.push_back(entry.path().stem().string());
            }
        }

        std::cout << "  Found " << frame_ids.size() << " frames" << std::endl;

        // Compute ray directions once per scene (intrinsics are constant)
        std::cout << "  Computing ray directions..." << std::endl;
        Eigen::MatrixXf rays = computer.computeRayDirections(K, height, width);

        // Save ray directions for each frame
        int frames_saved = 0;
        for (const auto& frame_id : frame_ids) {
            std::string ray_path = rays_dir + "/" + frame_id + ".bin";

            if (computer.saveRayDirections(rays, height, width, ray_path)) {
                frames_saved++;
            }
        }

        std::cout << "  Saved ray directions for " << frames_saved << " frames" << std::endl;

        scenes_processed++;
        total_frames += frames_saved;
    }

    std::cout << "\n=== Preprocessing Complete ===" << std::endl;
    std::cout << "Scenes processed: " << scenes_processed << std::endl;
    std::cout << "Total frames: " << total_frames << std::endl;
    std::cout << std::endl;

    return 0;
}
