#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct ValidationResult {
    std::string image_path;
    std::string sensor_type;  // kv1, kv2, realsense, xtion
    bool has_intrinsics;
    bool has_rgb;
    bool has_depth;
    bool has_extrinsics;
    bool has_scene_info;
    std::vector<std::string> errors;
    bool is_valid;

    // Image properties
    int rgb_width;
    int rgb_height;
    int depth_width;
    int depth_height;
};

class SunRGBDValidator {
public:
    SunRGBDValidator(const std::string& data_dir) : data_dir_(data_dir) {}

    std::vector<ValidationResult> validateAllImages() {
        std::vector<ValidationResult> results;

        if (!fs::exists(data_dir_)) {
            std::cerr << "Error: Data directory does not exist: " << data_dir_ << std::endl;
            return results;
        }

        // Check for SUNRGBD directory
        std::string sunrgbd_dir = data_dir_ + "/SUNRGBD";
        if (!fs::exists(sunrgbd_dir)) {
            std::cerr << "Error: SUNRGBD directory not found: " << sunrgbd_dir << std::endl;
            return results;
        }

        // Sensor types in SUN RGB-D
        std::vector<std::string> sensor_types = {"kv1", "kv2", "realsense", "xtion"};

        for (const auto& sensor : sensor_types) {
            std::string sensor_dir = sunrgbd_dir + "/" + sensor;
            if (fs::exists(sensor_dir) && fs::is_directory(sensor_dir)) {
                std::cout << "\n=== Validating " << sensor << " sensor data ===" << std::endl;
                validateSensorDirectory(sensor_dir, sensor, results);
            } else {
                std::cout << "Sensor directory not found: " << sensor << std::endl;
            }
        }

        return results;
    }

private:
    std::string data_dir_;

    void validateSensorDirectory(const std::string& sensor_dir,
                                 const std::string& sensor_type,
                                 std::vector<ValidationResult>& results) {
        int valid_count = 0;
        int total_count = 0;

        // Recursively find all image directories
        for (const auto& entry : fs::recursive_directory_iterator(sensor_dir)) {
            if (entry.is_directory()) {
                std::string dir_path = entry.path().string();

                // Check if this directory contains an image
                std::string image_path = dir_path + "/image";
                if (fs::exists(image_path) && fs::is_directory(image_path)) {
                    total_count++;
                    ValidationResult result = validateImageDirectory(dir_path, sensor_type);
                    results.push_back(result);

                    if (result.is_valid) {
                        valid_count++;
                    }

                    if (total_count % 100 == 0) {
                        std::cout << "  Processed " << total_count << " images..." << std::endl;
                    }
                }
            }
        }

        std::cout << "  Total images found: " << total_count << std::endl;
        std::cout << "  Valid images: " << valid_count << std::endl;
    }

    ValidationResult validateImageDirectory(const std::string& dir_path,
                                           const std::string& sensor_type) {
        ValidationResult result;
        result.image_path = dir_path;
        result.sensor_type = sensor_type;
        result.has_intrinsics = false;
        result.has_rgb = false;
        result.has_depth = false;
        result.has_extrinsics = false;
        result.has_scene_info = false;
        result.is_valid = false;
        result.rgb_width = 0;
        result.rgb_height = 0;
        result.depth_width = 0;
        result.depth_height = 0;

        // Check for intrinsics
        std::string intrinsics_path = dir_path + "/intrinsics.txt";
        if (fs::exists(intrinsics_path)) {
            result.has_intrinsics = validateIntrinsics(intrinsics_path, result.errors);
        } else {
            result.errors.push_back("Missing intrinsics.txt");
        }

        // Check for RGB image
        std::string image_dir = dir_path + "/image";
        if (fs::exists(image_dir) && fs::is_directory(image_dir)) {
            result.has_rgb = validateRGBImage(image_dir, result);
        } else {
            result.errors.push_back("Missing image directory");
        }

        // Check for depth image
        std::string depth_dir = dir_path + "/depth";
        if (fs::exists(depth_dir) && fs::is_directory(depth_dir)) {
            result.has_depth = validateDepthImage(depth_dir, result);
        } else {
            result.errors.push_back("Missing depth directory");
        }

        // Check for extrinsics (optional in some cases)
        std::string extrinsics_dir = dir_path + "/extrinsics";
        if (fs::exists(extrinsics_dir) && fs::is_directory(extrinsics_dir)) {
            result.has_extrinsics = true;
        }

        // Check for scene info
        std::string scene_path = dir_path + "/scene.txt";
        if (fs::exists(scene_path)) {
            result.has_scene_info = true;
        }

        // Overall validation
        result.is_valid = result.has_intrinsics && result.has_rgb && result.has_depth;

        return result;
    }

    bool validateIntrinsics(const std::string& filepath, std::vector<std::string>& errors) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            errors.push_back("Cannot open intrinsics file: " + filepath);
            return false;
        }

        Eigen::Matrix3f K;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (!(file >> K(i, j))) {
                    errors.push_back("Invalid intrinsics format");
                    return false;
                }
            }
        }

        // Validate matrix structure
        if (K(0, 0) <= 0 || K(1, 1) <= 0) {
            errors.push_back("Invalid focal lengths");
            return false;
        }

        if (std::abs(K(2, 2) - 1.0f) > 1e-6) {
            errors.push_back("Invalid intrinsics K(2,2) should be 1");
            return false;
        }

        return true;
    }

    bool validateRGBImage(const std::string& image_dir, ValidationResult& result) {
        // Find first .jpg or .png file in directory
        for (const auto& entry : fs::directory_iterator(image_dir)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                cv::Mat img = cv::imread(entry.path().string());
                if (img.empty()) {
                    result.errors.push_back("Cannot load RGB image");
                    return false;
                }

                result.rgb_height = img.rows;
                result.rgb_width = img.cols;
                return true;
            }
        }

        result.errors.push_back("No RGB image found in directory");
        return false;
    }

    bool validateDepthImage(const std::string& depth_dir, ValidationResult& result) {
        // Find first .png file in directory
        for (const auto& entry : fs::directory_iterator(depth_dir)) {
            if (entry.path().extension() == ".png") {
                cv::Mat depth = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
                if (depth.empty()) {
                    result.errors.push_back("Cannot load depth image");
                    return false;
                }

                result.depth_height = depth.rows;
                result.depth_width = depth.cols;

                // Check depth format
                if (depth.type() != CV_16UC1 && depth.type() != CV_32FC1) {
                    result.errors.push_back("Invalid depth format (expected 16-bit or 32-bit)");
                    return false;
                }

                return true;
            }
        }

        result.errors.push_back("No depth image found in directory");
        return false;
    }
};

void printSummary(const std::vector<ValidationResult>& results) {
    std::cout << "\n\n=== VALIDATION SUMMARY ===" << std::endl;
    std::cout << "Total images validated: " << results.size() << std::endl;

    int valid_count = 0;
    std::map<std::string, int> sensor_counts;
    std::map<std::string, int> sensor_valid_counts;

    for (const auto& result : results) {
        sensor_counts[result.sensor_type]++;
        if (result.is_valid) {
            valid_count++;
            sensor_valid_counts[result.sensor_type]++;
        }
    }

    std::cout << "Valid images: " << valid_count << "/" << results.size() << std::endl;
    std::cout << "\nBreakdown by sensor:" << std::endl;

    for (const auto& [sensor, count] : sensor_counts) {
        int valid = sensor_valid_counts[sensor];
        std::cout << "  " << sensor << ": " << valid << "/" << count << " valid" << std::endl;
    }

    if (valid_count < results.size()) {
        std::cout << "\n=== SAMPLE ERRORS (first 10) ===" << std::endl;
        int error_count = 0;
        for (const auto& result : results) {
            if (!result.is_valid && error_count < 10) {
                std::cout << "\nImage: " << result.image_path << std::endl;
                std::cout << "Sensor: " << result.sensor_type << std::endl;
                for (const auto& error : result.errors) {
                    std::cout << "  - " << error << std::endl;
                }
                error_count++;
            }
        }
    }
}

void saveManifest(const std::vector<ValidationResult>& results, const std::string& output_path) {
    json manifest;
    manifest["dataset"] = "SUN RGB-D V1";
    manifest["total_images"] = results.size();

    int valid_count = 0;
    json images_array = json::array();
    std::map<std::string, int> sensor_counts;

    for (const auto& result : results) {
        sensor_counts[result.sensor_type]++;

        if (result.is_valid) {
            valid_count++;

            json image_info;
            image_info["path"] = result.image_path;
            image_info["sensor_type"] = result.sensor_type;
            image_info["has_intrinsics"] = result.has_intrinsics;
            image_info["has_rgb"] = result.has_rgb;
            image_info["has_depth"] = result.has_depth;
            image_info["has_extrinsics"] = result.has_extrinsics;
            image_info["has_scene_info"] = result.has_scene_info;
            image_info["rgb_resolution"] = {result.rgb_width, result.rgb_height};
            image_info["depth_resolution"] = {result.depth_width, result.depth_height};
            image_info["valid"] = result.is_valid;

            images_array.push_back(image_info);
        }
    }

    manifest["valid_images"] = valid_count;
    manifest["sensor_counts"] = sensor_counts;
    manifest["images"] = images_array;

    std::ofstream file(output_path);
    file << manifest.dump(2);  // Pretty print with 2-space indent
    file.close();

    std::cout << "\nManifest saved to: " << output_path << std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir = "./data/sunrgbd";
    std::string manifest_path = "./data/sunrgbd_manifest.json";

    if (argc > 1) {
        data_dir = argv[1];
    }

    if (argc > 2) {
        manifest_path = argv[2];
    }

    std::cout << "=== SUN RGB-D Data Validation Tool ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << "Manifest output: " << manifest_path << std::endl;

    SunRGBDValidator validator(data_dir);
    std::vector<ValidationResult> results = validator.validateAllImages();

    printSummary(results);
    saveManifest(results, manifest_path);

    std::cout << "\n=== Validation complete ===" << std::endl;

    return 0;
}
