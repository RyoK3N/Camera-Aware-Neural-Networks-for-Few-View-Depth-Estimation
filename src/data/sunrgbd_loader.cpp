#include "sunrgbd_loader.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace camera_aware_depth {

SunRGBDLoader::SunRGBDLoader(
    const std::string& data_dir,
    const std::string& manifest_path,
    const std::string& split
) : data_dir_(data_dir),
    manifest_path_(manifest_path),
    split_(split),
    augmentation_enabled_(false),
    target_height_(480),
    target_width_(640),
    rng_(42)
{
    // Default: allow all sensor types
    allowed_sensors_ = {"kv1", "kv2", "realsense", "xtion"};

    loadManifest();

    std::cout << "SunRGBDLoader initialized:" << std::endl;
    std::cout << "  Split: " << split_ << std::endl;
    std::cout << "  Samples: " << sample_paths_.size() << std::endl;
}

SunRGBDLoader::~SunRGBDLoader() {
    // Destructor
}

void SunRGBDLoader::loadManifest() {
    std::ifstream file(manifest_path_);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open manifest file: " + manifest_path_);
    }

    json manifest;
    file >> manifest;
    file.close();

    // Extract image paths for the specified split
    for (const auto& image_info : manifest["images"]) {
        if (!image_info["valid"]) continue;

        std::string sensor_type = image_info["sensor_type"];

        // Filter by allowed sensor types
        if (std::find(allowed_sensors_.begin(), allowed_sensors_.end(), sensor_type) == allowed_sensors_.end()) {
            continue;
        }

        SamplePath sample;
        sample.image_dir = image_info["path"];
        sample.sensor_type = sensor_type;

        // Construct file paths based on SUN RGB-D structure
        sample.rgb_path = "";  // Will be found dynamically
        sample.depth_path = "";  // Will be found dynamically
        sample.intrinsic_path = sample.image_dir + "/intrinsics.txt";
        sample.extrinsic_path = sample.image_dir + "/extrinsics";
        sample.scene_path = sample.image_dir + "/scene.txt";
        sample.ray_path = sample.image_dir + "/rays.bin";

        // Verify required files exist
        if (fs::exists(sample.intrinsic_path)) {
            sample_paths_.push_back(sample);
        }
    }

    std::cout << "Loaded " << sample_paths_.size() << " samples from manifest" << std::endl;
}

std::string SunRGBDLoader::findRGBImage(const std::string& image_dir) {
    std::string rgb_dir = image_dir + "/image";
    if (!fs::exists(rgb_dir)) return "";

    for (const auto& entry : fs::directory_iterator(rgb_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            return entry.path().string();
        }
    }
    return "";
}

std::string SunRGBDLoader::findDepthImage(const std::string& depth_dir) {
    std::string depth_path = depth_dir + "/depth";
    if (!fs::exists(depth_path)) return "";

    for (const auto& entry : fs::directory_iterator(depth_path)) {
        if (entry.path().extension() == ".png") {
            return entry.path().string();
        }
    }
    return "";
}

SunRGBDSample SunRGBDLoader::getSample(size_t index) {
    if (index >= sample_paths_.size()) {
        throw std::out_of_range("Sample index out of range");
    }

    const SamplePath& path = sample_paths_[index];

    SunRGBDSample sample;
    sample.image_path = path.image_dir;
    sample.sensor_type = path.sensor_type;

    // Find and load RGB
    std::string rgb_path = findRGBImage(path.image_dir);
    if (!rgb_path.empty()) {
        sample.rgb = loadRGB(rgb_path);
    } else {
        throw std::runtime_error("RGB image not found: " + path.image_dir);
    }

    // Find and load depth
    std::string depth_path = findDepthImage(path.image_dir);
    if (!depth_path.empty()) {
        sample.depth = loadDepth(depth_path);
    } else {
        throw std::runtime_error("Depth image not found: " + path.image_dir);
    }

    // Load intrinsics
    sample.intrinsics = loadIntrinsics(path.intrinsic_path);

    // Load extrinsics (optional)
    if (fs::exists(path.extrinsic_path)) {
        sample.extrinsics = loadExtrinsics(path.extrinsic_path);
    } else {
        // Create identity matrix if not available
        sample.extrinsics = torch::eye(4, torch::kFloat32);
    }

    // Load scene type
    if (fs::exists(path.scene_path)) {
        sample.scene_type = loadSceneType(path.scene_path);
    }

    // Load or compute ray directions
    if (fs::exists(path.ray_path)) {
        sample.ray_directions = loadRayDirections(path.ray_path);
    } else {
        // Ray directions will need to be computed
        int H = sample.rgb.size(1);
        int W = sample.rgb.size(2);
        sample.ray_directions = torch::zeros({3, H, W});
    }

    // Resize to target dimensions
    resizeSample(sample);

    // Apply augmentation if enabled
    if (augmentation_enabled_ && split_ == "train") {
        sample = augmentSample(sample);
    }

    return sample;
}

std::vector<SunRGBDSample> SunRGBDLoader::getBatch(const std::vector<size_t>& indices) {
    std::vector<SunRGBDSample> batch;
    batch.reserve(indices.size());

    for (size_t idx : indices) {
        batch.push_back(getSample(idx));
    }

    return batch;
}

void SunRGBDLoader::enableAugmentation(const AugmentationConfig& config) {
    augmentation_enabled_ = true;
    aug_config_ = config;
    rng_.seed(config.random_seed);
}

void SunRGBDLoader::disableAugmentation() {
    augmentation_enabled_ = false;
}

void SunRGBDLoader::setTargetDimensions(int height, int width) {
    target_height_ = height;
    target_width_ = width;
}

void SunRGBDLoader::filterBySensorType(const std::vector<std::string>& sensor_types) {
    allowed_sensors_ = sensor_types;
    // Reload manifest with new filter
    sample_paths_.clear();
    loadManifest();
}

std::string SunRGBDLoader::getStatistics() const {
    std::stringstream ss;
    ss << "SUN RGB-D Loader Statistics:" << std::endl;
    ss << "  Split: " << split_ << std::endl;
    ss << "  Total samples: " << sample_paths_.size() << std::endl;
    ss << "  Target dimensions: " << target_height_ << "x" << target_width_ << std::endl;
    ss << "  Augmentation: " << (augmentation_enabled_ ? "enabled" : "disabled") << std::endl;
    ss << "  Allowed sensors: ";
    for (const auto& sensor : allowed_sensors_) {
        ss << sensor << " ";
    }
    ss << std::endl;
    return ss.str();
}

// Private methods

torch::Tensor SunRGBDLoader::loadRGB(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Cannot load RGB image: " + path);
    }

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert to tensor and normalize
    torch::Tensor tensor = matToTensor(img);
    return normalizeRGB(tensor);
}

torch::Tensor SunRGBDLoader::loadDepth(const std::string& path) {
    cv::Mat depth = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
        throw std::runtime_error("Cannot load depth map: " + path);
    }

    // Convert to float (SUN RGB-D depth format varies by sensor)
    if (depth.type() == CV_16UC1) {
        depth.convertTo(depth, CV_32F, 1.0 / 1000.0); // Convert to meters
    } else if (depth.type() == CV_32FC1) {
        // Already in float format
    } else {
        depth.convertTo(depth, CV_32F);
    }

    // Convert to tensor (1, H, W)
    torch::Tensor tensor = matToTensor(depth);
    if (tensor.dim() == 3 && tensor.size(0) == 1) {
        return tensor;
    } else if (tensor.dim() == 2) {
        return tensor.unsqueeze(0);
    }

    return tensor;
}

torch::Tensor SunRGBDLoader::loadIntrinsics(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open intrinsics file: " + path);
    }

    std::vector<float> values(9);
    for (int i = 0; i < 9; ++i) {
        file >> values[i];
    }
    file.close();

    // Create 3x3 tensor
    return torch::from_blob(values.data(), {3, 3}, torch::kFloat32).clone();
}

torch::Tensor SunRGBDLoader::loadExtrinsics(const std::string& path) {
    // SUN RGB-D extrinsics are in a different format
    // Typically rotation matrix to align with gravity
    // This is a simplified version - adjust based on actual format

    if (!fs::is_directory(path)) {
        return torch::eye(4, torch::kFloat32);
    }

    // Look for rotation matrix file
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream file(entry.path().string());
            if (!file.is_open()) continue;

            // Try to read 3x3 or 4x4 matrix
            std::vector<float> values(16);
            int count = 0;
            while (count < 16 && file >> values[count]) {
                count++;
            }
            file.close();

            if (count == 9) {
                // 3x3 rotation matrix - extend to 4x4
                auto R = torch::from_blob(values.data(), {3, 3}, torch::kFloat32).clone();
                auto pose = torch::eye(4, torch::kFloat32);
                pose.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R);
                return pose;
            } else if (count == 16) {
                // 4x4 transformation matrix
                return torch::from_blob(values.data(), {4, 4}, torch::kFloat32).clone();
            }
        }
    }

    return torch::eye(4, torch::kFloat32);
}

std::string SunRGBDLoader::loadSceneType(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "unknown";
    }

    std::string scene_type;
    std::getline(file, scene_type);
    file.close();

    return scene_type;
}

torch::Tensor SunRGBDLoader::loadRayDirections(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ray directions file: " + path);
    }

    // Read header
    int32_t height, width;
    file.read(reinterpret_cast<char*>(&height), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&width), sizeof(int32_t));

    // Read ray data
    std::vector<float> ray_data(height * width * 3);
    file.read(reinterpret_cast<char*>(ray_data.data()), ray_data.size() * sizeof(float));
    file.close();

    // Create tensor (H*W, 3) and reshape to (3, H, W)
    torch::Tensor rays = torch::from_blob(ray_data.data(), {height * width, 3}, torch::kFloat32).clone();
    rays = rays.view({height, width, 3}).permute({2, 0, 1}); // (3, H, W)

    return rays;
}

SunRGBDSample SunRGBDLoader::augmentSample(SunRGBDSample sample) {
    // Random crop
    if (aug_config_.enable_random_crop) {
        std::uniform_real_distribution<float> scale_dist(
            aug_config_.crop_scale_min, aug_config_.crop_scale_max);
        float scale = scale_dist(rng_);

        int H = sample.rgb.size(1);
        int W = sample.rgb.size(2);
        int crop_h = static_cast<int>(H * scale);
        int crop_w = static_cast<int>(W * scale);

        std::uniform_int_distribution<int> x_dist(0, std::max(1, W - crop_w));
        std::uniform_int_distribution<int> y_dist(0, std::max(1, H - crop_h));

        int crop_x = x_dist(rng_);
        int crop_y = y_dist(rng_);

        applyCrop(sample, scale, crop_x, crop_y);
    }

    // Horizontal flip
    if (aug_config_.enable_horizontal_flip) {
        std::uniform_real_distribution<float> flip_dist(0.0f, 1.0f);
        if (flip_dist(rng_) < aug_config_.horizontal_flip_prob) {
            applyHorizontalFlip(sample);
        }
    }

    // Color jitter
    if (aug_config_.enable_color_jitter) {
        applyColorJitter(sample);
    }

    return sample;
}

void SunRGBDLoader::applyCrop(SunRGBDSample& sample, float scale, int crop_x, int crop_y) {
    int H = sample.rgb.size(1);
    int W = sample.rgb.size(2);
    int crop_h = static_cast<int>(H * scale);
    int crop_w = static_cast<int>(W * scale);

    // Crop tensors
    sample.rgb = sample.rgb.index({torch::indexing::Slice(),
                                   torch::indexing::Slice(crop_y, crop_y + crop_h),
                                   torch::indexing::Slice(crop_x, crop_x + crop_w)});

    sample.depth = sample.depth.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(crop_y, crop_y + crop_h),
                                       torch::indexing::Slice(crop_x, crop_x + crop_w)});

    sample.ray_directions = sample.ray_directions.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(crop_y, crop_y + crop_h),
        torch::indexing::Slice(crop_x, crop_x + crop_w)});

    // Update intrinsics (adjust principal point)
    auto K = sample.intrinsics.clone();
    K[0][2] = K[0][2].item<float>() - crop_x; // cx
    K[1][2] = K[1][2].item<float>() - crop_y; // cy
    sample.intrinsics = K;
}

void SunRGBDLoader::applyHorizontalFlip(SunRGBDSample& sample) {
    // Flip tensors
    sample.rgb = torch::flip(sample.rgb, {2});
    sample.depth = torch::flip(sample.depth, {2});
    sample.ray_directions = torch::flip(sample.ray_directions, {2});

    // Flip x-component of ray directions
    sample.ray_directions[0] = -sample.ray_directions[0];

    // Update intrinsics (flip cx)
    auto K = sample.intrinsics.clone();
    int W = sample.rgb.size(2);
    K[0][2] = W - K[0][2].item<float>() - 1; // cx
    sample.intrinsics = K;
}

void SunRGBDLoader::applyColorJitter(SunRGBDSample& sample) {
    std::uniform_real_distribution<float> brightness_dist(
        1.0f - aug_config_.brightness_delta, 1.0f + aug_config_.brightness_delta);
    std::uniform_real_distribution<float> contrast_dist(
        1.0f - aug_config_.contrast_delta, 1.0f + aug_config_.contrast_delta);

    float brightness_factor = brightness_dist(rng_);
    float contrast_factor = contrast_dist(rng_);

    // Apply brightness and contrast
    sample.rgb = torch::clamp(sample.rgb * contrast_factor + brightness_factor - 1.0f, 0.0f, 1.0f);
}

void SunRGBDLoader::resizeSample(SunRGBDSample& sample) {
    int current_h = sample.rgb.size(1);
    int current_w = sample.rgb.size(2);

    if (current_h == target_height_ && current_w == target_width_) {
        return;
    }

    // Resize using interpolation
    sample.rgb = torch::nn::functional::interpolate(
        sample.rgb.unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_height_, target_width_})
            .mode(torch::kBilinear)
            .align_corners(false)
    ).squeeze(0);

    sample.depth = torch::nn::functional::interpolate(
        sample.depth.unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_height_, target_width_})
            .mode(torch::kNearest)
    ).squeeze(0);

    if (sample.ray_directions.numel() > 0) {
        sample.ray_directions = torch::nn::functional::interpolate(
            sample.ray_directions.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_height_, target_width_})
                .mode(torch::kBilinear)
                .align_corners(false)
        ).squeeze(0);
    }

    // Update intrinsics for new dimensions
    float scale_x = static_cast<float>(target_width_) / current_w;
    float scale_y = static_cast<float>(target_height_) / current_h;

    auto K = sample.intrinsics.clone();
    K[0][0] = K[0][0].item<float>() * scale_x; // fx
    K[1][1] = K[1][1].item<float>() * scale_y; // fy
    K[0][2] = K[0][2].item<float>() * scale_x; // cx
    K[1][2] = K[1][2].item<float>() * scale_y; // cy
    sample.intrinsics = K;
}

torch::Tensor SunRGBDLoader::matToTensor(const cv::Mat& mat) {
    // Convert OpenCV Mat to PyTorch Tensor
    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F);

    int height = float_mat.rows;
    int width = float_mat.cols;
    int channels = float_mat.channels();

    // Create tensor from data
    torch::Tensor tensor;
    if (channels == 1) {
        tensor = torch::from_blob(float_mat.data, {height, width}, torch::kFloat32).clone();
    } else {
        tensor = torch::from_blob(float_mat.data, {height, width, channels}, torch::kFloat32).clone();
        tensor = tensor.permute({2, 0, 1}); // HWC -> CHW
    }

    return tensor;
}

torch::Tensor SunRGBDLoader::normalizeRGB(torch::Tensor rgb) {
    // Normalize to [0, 1] range
    return rgb / 255.0f;
}

} // namespace camera_aware_depth
