#ifndef SUNRGBD_LOADER_H
#define SUNRGBD_LOADER_H

#include <torch/torch.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <random>

namespace camera_aware_depth {

/**
 * @brief Data structure representing a single SUN RGB-D sample
 */
struct SunRGBDSample {
    torch::Tensor rgb;              // RGB image (3, H, W)
    torch::Tensor depth;            // Depth map (1, H, W)
    torch::Tensor ray_directions;   // Ray directions (3, H, W)
    torch::Tensor intrinsics;       // Camera intrinsics (3, 3)
    torch::Tensor extrinsics;       // Camera extrinsics (4, 4) - optional
    std::string image_path;
    std::string sensor_type;        // kv1, kv2, realsense, or xtion
    std::string scene_type;         // Scene classification
};

/**
 * @brief Configuration for data augmentation
 */
struct AugmentationConfig {
    bool enable_random_crop = true;
    float crop_scale_min = 0.7f;
    float crop_scale_max = 1.0f;

    bool enable_horizontal_flip = true;
    float horizontal_flip_prob = 0.5f;

    bool enable_color_jitter = true;
    float brightness_delta = 0.2f;
    float contrast_delta = 0.2f;
    float saturation_delta = 0.2f;
    float hue_delta = 0.1f;

    int random_seed = 42;
};

/**
 * @brief SUN RGB-D Dataset Loader
 *
 * Loads RGB images, depth maps, camera parameters, and ray directions
 * from the SUN RGB-D dataset (10,335 images from NYU Depth v2, Berkeley B3DO, and SUN3D).
 *
 * Dataset Features:
 * - 4 different sensors (Kinect v1, v2, RealSense, Xtion)
 * - Indoor scenes
 * - Camera intrinsics and extrinsics
 * - Scene type information
 */
class SunRGBDLoader {
public:
    /**
     * @brief Construct a new SUN RGB-D Loader object
     *
     * @param data_dir Root directory containing SUN RGB-D data
     * @param manifest_path Path to data manifest JSON file
     * @param split Dataset split ("train", "test")
     */
    SunRGBDLoader(
        const std::string& data_dir,
        const std::string& manifest_path,
        const std::string& split
    );

    /**
     * @brief Destroy the SUN RGB-D Loader object
     */
    ~SunRGBDLoader();

    /**
     * @brief Get a single sample by index
     *
     * @param index Sample index
     * @return SunRGBDSample The loaded sample
     */
    SunRGBDSample getSample(size_t index);

    /**
     * @brief Get a batch of samples
     *
     * @param indices Vector of sample indices
     * @return std::vector<SunRGBDSample> Vector of loaded samples
     */
    std::vector<SunRGBDSample> getBatch(const std::vector<size_t>& indices);

    /**
     * @brief Get total number of samples
     *
     * @return size_t Number of samples
     */
    size_t size() const { return sample_paths_.size(); }

    /**
     * @brief Enable data augmentation
     *
     * @param config Augmentation configuration
     */
    void enableAugmentation(const AugmentationConfig& config);

    /**
     * @brief Disable data augmentation
     */
    void disableAugmentation();

    /**
     * @brief Set target image dimensions
     *
     * @param height Target height
     * @param width Target width
     */
    void setTargetDimensions(int height, int width);

    /**
     * @brief Filter by sensor type
     *
     * @param sensor_types Vector of sensor types to include (kv1, kv2, realsense, xtion)
     */
    void filterBySensorType(const std::vector<std::string>& sensor_types);

    /**
     * @brief Get dataset statistics
     *
     * @return std::string Statistics as formatted string
     */
    std::string getStatistics() const;

private:
    struct SamplePath {
        std::string image_dir;          // Directory containing image/, depth/, etc.
        std::string sensor_type;
        std::string rgb_path;
        std::string depth_path;
        std::string intrinsic_path;
        std::string extrinsic_path;
        std::string scene_path;
        std::string ray_path;           // Precomputed rays (if available)
    };

    std::string data_dir_;
    std::string manifest_path_;
    std::string split_;

    std::vector<SamplePath> sample_paths_;
    std::vector<std::string> allowed_sensors_;

    bool augmentation_enabled_;
    AugmentationConfig aug_config_;

    int target_height_;
    int target_width_;

    std::mt19937 rng_;

    /**
     * @brief Load manifest and build sample paths
     */
    void loadManifest();

    /**
     * @brief Find RGB image in image directory
     */
    std::string findRGBImage(const std::string& image_dir);

    /**
     * @brief Find depth image in depth directory
     */
    std::string findDepthImage(const std::string& depth_dir);

    /**
     * @brief Load RGB image from file
     */
    torch::Tensor loadRGB(const std::string& path);

    /**
     * @brief Load depth map from file
     */
    torch::Tensor loadDepth(const std::string& path);

    /**
     * @brief Load camera intrinsics from file
     */
    torch::Tensor loadIntrinsics(const std::string& path);

    /**
     * @brief Load camera extrinsics from file (optional)
     */
    torch::Tensor loadExtrinsics(const std::string& path);

    /**
     * @brief Load scene type from file
     */
    std::string loadSceneType(const std::string& path);

    /**
     * @brief Load precomputed ray directions from file
     */
    torch::Tensor loadRayDirections(const std::string& path);

    /**
     * @brief Apply data augmentation to a sample
     */
    SunRGBDSample augmentSample(SunRGBDSample sample);

    /**
     * @brief Apply random crop augmentation
     */
    void applyCrop(SunRGBDSample& sample, float scale, int crop_x, int crop_y);

    /**
     * @brief Apply horizontal flip augmentation
     */
    void applyHorizontalFlip(SunRGBDSample& sample);

    /**
     * @brief Apply color jittering augmentation
     */
    void applyColorJitter(SunRGBDSample& sample);

    /**
     * @brief Resize sample to target dimensions
     */
    void resizeSample(SunRGBDSample& sample);

    /**
     * @brief Convert OpenCV Mat to PyTorch Tensor
     */
    torch::Tensor matToTensor(const cv::Mat& mat);

    /**
     * @brief Normalize RGB image to [0, 1] range
     */
    torch::Tensor normalizeRGB(torch::Tensor rgb);
};

/**
 * @brief PyTorch DataLoader compatible dataset wrapper
 */
class SunRGBDDataset : public torch::data::Dataset<SunRGBDDataset, SunRGBDSample> {
public:
    SunRGBDDataset(std::shared_ptr<SunRGBDLoader> loader)
        : loader_(loader) {}

    SunRGBDSample get(size_t index) override {
        return loader_->getSample(index);
    }

    torch::optional<size_t> size() const override {
        return loader_->size();
    }

private:
    std::shared_ptr<SunRGBDLoader> loader_;
};

} // namespace camera_aware_depth

#endif // SUNRGBD_LOADER_H
