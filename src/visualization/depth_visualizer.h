#ifndef DEPTH_VISUALIZER_H
#define DEPTH_VISUALIZER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace camera_aware_depth {

/**
 * @brief Colormap types for depth visualization
 */
enum class ColormapType {
    VIRIDIS,      // Purple-green-yellow (perceptually uniform)
    PLASMA,       // Purple-red-yellow (perceptually uniform)
    MAGMA,        // Black-purple-yellow (perceptually uniform)
    INFERNO,      // Black-red-yellow (perceptually uniform)
    TURBO,        // Rainbow-like but better
    JET,          // Traditional rainbow (not recommended)
    GRAY,         // Grayscale
    HOT,          // Black-red-yellow-white
    COOL          // Cyan-magenta
};

/**
 * @brief Depth Visualizer
 *
 * Production-quality depth visualization with:
 * - Multiple perceptually uniform colormaps
 * - Error map visualization
 * - Side-by-side comparisons
 * - Depth histograms
 * - 3D point cloud rendering (optional)
 *
 * Based on best practices from:
 * - MiDaS visualization (https://github.com/isl-org/MiDaS)
 * - NYU Depth V2 visualization tools
 * - Matplotlib perceptually uniform colormaps
 */
class DepthVisualizer {
public:
    /**
     * @brief Apply colormap to depth map
     *
     * @param depth Depth map (H, W) - single channel
     * @param min_depth Minimum depth value for normalization
     * @param max_depth Maximum depth value for normalization
     * @param colormap Colormap type
     * @return RGB image (H, W, 3) in [0, 255]
     */
    static cv::Mat applyColormap(
        const torch::Tensor& depth,
        float min_depth,
        float max_depth,
        ColormapType colormap = ColormapType::VIRIDIS
    ) {
        // Convert to OpenCV Mat
        auto depth_cpu = depth.cpu();
        cv::Mat depth_mat(depth_cpu.size(0), depth_cpu.size(1), CV_32F, depth_cpu.data_ptr<float>());

        // Normalize to [0, 1]
        cv::Mat normalized;
        cv::normalize(depth_mat, normalized, 0.0, 1.0, cv::NORM_MINMAX);

        // Clip to valid range
        normalized = (normalized - min_depth) / (max_depth - min_depth);
        normalized = cv::max(cv::min(normalized, 1.0), 0.0);

        // Convert to 8-bit
        cv::Mat depth_8u;
        normalized.convertTo(depth_8u, CV_8U, 255.0);

        // Apply colormap
        cv::Mat colored;
        int cv_colormap = getOpenCVColormap(colormap);

        if (cv_colormap >= 0) {
            cv::applyColorMap(depth_8u, colored, cv_colormap);
        } else {
            // Custom colormaps
            colored = applyCustomColormap(depth_8u, colormap);
        }

        return colored;
    }

    /**
     * @brief Create error map visualization
     *
     * @param pred_depth Predicted depth (H, W)
     * @param gt_depth Ground truth depth (H, W)
     * @param max_error Maximum error for color scaling (default: 1.0m)
     * @return RGB error map (H, W, 3)
     */
    static cv::Mat createErrorMap(
        const torch::Tensor& pred_depth,
        const torch::Tensor& gt_depth,
        float max_error = 1.0f
    ) {
        // Compute absolute error
        auto error = torch::abs(pred_depth - gt_depth);

        // Convert to OpenCV
        auto error_cpu = error.cpu();
        cv::Mat error_mat(error_cpu.size(0), error_cpu.size(1), CV_32F, error_cpu.data_ptr<float>());

        // Normalize to [0, 1]
        error_mat = cv::min(error_mat / max_error, 1.0);

        // Convert to 8-bit
        cv::Mat error_8u;
        error_mat.convertTo(error_8u, CV_8U, 255.0);

        // Apply hot colormap (black -> red -> yellow -> white)
        cv::Mat colored;
        cv::applyColorMap(error_8u, colored, cv::COLORMAP_HOT);

        return colored;
    }

    /**
     * @brief Create side-by-side comparison
     *
     * @param rgb Input RGB image (3, H, W)
     * @param pred_depth Predicted depth (H, W)
     * @param gt_depth Ground truth depth (H, W)
     * @param min_depth Min depth for visualization
     * @param max_depth Max depth for visualization
     * @return Comparison image (H, W*4, 3) - [RGB | GT | Pred | Error]
     */
    static cv::Mat createComparison(
        const torch::Tensor& rgb,
        const torch::Tensor& pred_depth,
        const torch::Tensor& gt_depth,
        float min_depth,
        float max_depth
    ) {
        // Convert RGB to OpenCV (CHW -> HWC)
        auto rgb_hwc = rgb.permute({1, 2, 0}).contiguous();
        auto rgb_cpu = rgb_hwc.cpu();
        cv::Mat rgb_mat(rgb_cpu.size(0), rgb_cpu.size(1), CV_32FC3, rgb_cpu.data_ptr<float>());

        // Convert to 8-bit and BGR
        cv::Mat rgb_8u;
        rgb_mat.convertTo(rgb_8u, CV_8UC3, 255.0);
        cv::cvtColor(rgb_8u, rgb_8u, cv::COLOR_RGB2BGR);

        // Create depth visualizations
        cv::Mat gt_vis = applyColormap(gt_depth, min_depth, max_depth, ColormapType::VIRIDIS);
        cv::Mat pred_vis = applyColormap(pred_depth, min_depth, max_depth, ColormapType::VIRIDIS);
        cv::Mat error_vis = createErrorMap(pred_depth, gt_depth, 1.0f);

        // Add labels
        addLabel(rgb_8u, "Input RGB");
        addLabel(gt_vis, "Ground Truth");
        addLabel(pred_vis, "Prediction");
        addLabel(error_vis, "Error Map");

        // Concatenate horizontally
        cv::Mat result;
        cv::hconcat(std::vector<cv::Mat>{rgb_8u, gt_vis, pred_vis, error_vis}, result);

        return result;
    }

    /**
     * @brief Create depth histogram
     *
     * @param depth Depth map (H, W)
     * @param min_depth Min depth value
     * @param max_depth Max depth value
     * @param num_bins Number of histogram bins
     * @return Histogram image (H=300, W=512, 3)
     */
    static cv::Mat createHistogram(
        const torch::Tensor& depth,
        float min_depth,
        float max_depth,
        int num_bins = 50
    ) {
        // Convert to vector
        auto depth_flat = depth.flatten();
        auto depth_cpu = depth_flat.cpu();
        std::vector<float> depth_vec(depth_cpu.data_ptr<float>(),
                                     depth_cpu.data_ptr<float>() + depth_cpu.numel());

        // Filter valid depths
        std::vector<float> valid_depths;
        for (float d : depth_vec) {
            if (d >= min_depth && d <= max_depth) {
                valid_depths.push_back(d);
            }
        }

        if (valid_depths.empty()) {
            return cv::Mat(300, 512, CV_8UC3, cv::Scalar(255, 255, 255));
        }

        // Compute histogram
        std::vector<int> hist(num_bins, 0);
        float bin_width = (max_depth - min_depth) / num_bins;

        for (float d : valid_depths) {
            int bin = static_cast<int>((d - min_depth) / bin_width);
            bin = std::min(bin, num_bins - 1);
            hist[bin]++;
        }

        // Find max count for normalization
        int max_count = *std::max_element(hist.begin(), hist.end());

        // Create image
        int img_width = 512;
        int img_height = 300;
        cv::Mat hist_img(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw histogram
        int bar_width = img_width / num_bins;

        for (int i = 0; i < num_bins; ++i) {
            int bar_height = static_cast<int>(
                static_cast<float>(hist[i]) / max_count * (img_height - 50)
            );

            cv::Point pt1(i * bar_width, img_height - 30 - bar_height);
            cv::Point pt2((i + 1) * bar_width - 1, img_height - 30);

            cv::rectangle(hist_img, pt1, pt2, cv::Scalar(70, 130, 180), -1);
            cv::rectangle(hist_img, pt1, pt2, cv::Scalar(0, 0, 0), 1);
        }

        // Add axes
        cv::line(hist_img, cv::Point(0, img_height - 30),
                cv::Point(img_width, img_height - 30), cv::Scalar(0, 0, 0), 2);

        // Add labels
        cv::putText(hist_img, "Depth Distribution",
                   cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                   0.7, cv::Scalar(0, 0, 0), 2);

        std::string min_label = std::to_string(min_depth) + "m";
        std::string max_label = std::to_string(max_depth) + "m";

        cv::putText(hist_img, min_label, cv::Point(5, img_height - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        cv::putText(hist_img, max_label, cv::Point(img_width - 50, img_height - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

        return hist_img;
    }

    /**
     * @brief Create comprehensive visualization with metrics
     *
     * Includes: RGB, GT, Pred, Error map, Histogram, Metrics text
     */
    static cv::Mat createComprehensiveVisualization(
        const torch::Tensor& rgb,
        const torch::Tensor& pred_depth,
        const torch::Tensor& gt_depth,
        const std::map<std::string, float>& metrics,
        float min_depth,
        float max_depth
    ) {
        // Create comparison (4 panels)
        cv::Mat comparison = createComparison(rgb, pred_depth, gt_depth, min_depth, max_depth);

        // Create histogram
        cv::Mat pred_hist = createHistogram(pred_depth, min_depth, max_depth);
        cv::Mat gt_hist = createHistogram(gt_depth, min_depth, max_depth);

        // Create metrics panel
        cv::Mat metrics_panel = createMetricsPanel(metrics, comparison.cols, 200);

        // Stack vertically
        cv::Mat top_row = comparison;
        cv::Mat middle_row;
        cv::hconcat(pred_hist, gt_hist, middle_row);

        // Resize middle row to match width
        cv::resize(middle_row, middle_row, cv::Size(top_row.cols, middle_row.rows));

        // Combine all
        cv::Mat result;
        cv::vconcat(std::vector<cv::Mat>{top_row, middle_row, metrics_panel}, result);

        return result;
    }

private:
    /**
     * @brief Get OpenCV colormap constant
     */
    static int getOpenCVColormap(ColormapType type) {
        switch (type) {
            case ColormapType::JET: return cv::COLORMAP_JET;
            case ColormapType::HOT: return cv::COLORMAP_HOT;
            case ColormapType::COOL: return cv::COLORMAP_COOL;
            case ColormapType::VIRIDIS: return cv::COLORMAP_VIRIDIS;
            case ColormapType::PLASMA: return cv::COLORMAP_PLASMA;
            case ColormapType::MAGMA: return cv::COLORMAP_MAGMA;
            case ColormapType::INFERNO: return cv::COLORMAP_INFERNO;
            case ColormapType::TURBO: return cv::COLORMAP_TURBO;
            default: return -1;
        }
    }

    /**
     * @brief Apply custom colormap
     */
    static cv::Mat applyCustomColormap(const cv::Mat& depth_8u, ColormapType type) {
        // For custom colormaps not in OpenCV
        // This is a placeholder - can implement custom color schemes
        cv::Mat colored;
        cv::applyColorMap(depth_8u, colored, cv::COLORMAP_VIRIDIS);
        return colored;
    }

    /**
     * @brief Add text label to image
     */
    static void addLabel(cv::Mat& img, const std::string& label) {
        cv::putText(img, label, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8,
                   cv::Scalar(255, 255, 255), 2);
        cv::putText(img, label, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8,
                   cv::Scalar(0, 0, 0), 1);
    }

    /**
     * @brief Create metrics display panel
     */
    static cv::Mat createMetricsPanel(
        const std::map<std::string, float>& metrics,
        int width,
        int height
    ) {
        cv::Mat panel(height, width, CV_8UC3, cv::Scalar(240, 240, 240));

        int y = 40;
        int line_height = 25;

        cv::putText(panel, "Metrics:", cv::Point(20, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        y += line_height + 10;

        // Display key metrics
        std::vector<std::pair<std::string, std::string>> display_metrics = {
            {"abs_rel", "AbsRel"},
            {"rmse", "RMSE"},
            {"rmse_log", "RMSElog"},
            {"delta_1.25", "δ < 1.25"}
        };

        for (const auto& [key, label] : display_metrics) {
            if (metrics.count(key)) {
                std::stringstream ss;
                ss << label << ": " << std::fixed << std::setprecision(4) << metrics.at(key);

                if (key.find("delta") != std::string::npos) {
                    ss << " (" << std::setprecision(2) << (metrics.at(key) * 100) << "%)";
                }

                cv::putText(panel, ss.str(), cv::Point(40, y),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                y += line_height;
            }
        }

        return panel;
    }
};

/**
 * @brief Batch Visualization Manager
 *
 * Efficiently manages visualization of multiple samples
 */
class BatchVisualizer {
public:
    /**
     * @brief Save batch of visualizations
     */
    static void saveVisualizationBatch(
        const std::vector<torch::Tensor>& rgbs,
        const std::vector<torch::Tensor>& pred_depths,
        const std::vector<torch::Tensor>& gt_depths,
        const std::vector<std::map<std::string, float>>& metrics,
        const std::string& output_dir,
        float min_depth,
        float max_depth
    ) {
        std::filesystem::create_directories(output_dir);

        for (size_t i = 0; i < rgbs.size(); ++i) {
            auto vis = DepthVisualizer::createComprehensiveVisualization(
                rgbs[i], pred_depths[i], gt_depths[i], metrics[i], min_depth, max_depth
            );

            std::string filename = output_dir + "/sample_" + std::to_string(i) + ".png";
            cv::imwrite(filename, vis);
        }
    }

    /**
     * @brief Create comparison grid
     *
     * Creates a grid of comparisons for multiple samples
     */
    static cv::Mat createComparisonGrid(
        const std::vector<torch::Tensor>& rgbs,
        const std::vector<torch::Tensor>& pred_depths,
        const std::vector<torch::Tensor>& gt_depths,
        int grid_cols,
        float min_depth,
        float max_depth
    ) {
        std::vector<cv::Mat> comparisons;

        for (size_t i = 0; i < rgbs.size(); ++i) {
            auto comp = DepthVisualizer::createComparison(
                rgbs[i], pred_depths[i], gt_depths[i], min_depth, max_depth
            );
            comparisons.push_back(comp);
        }

        // Arrange in grid
        int grid_rows = (comparisons.size() + grid_cols - 1) / grid_cols;
        std::vector<cv::Mat> rows;

        for (int r = 0; r < grid_rows; ++r) {
            std::vector<cv::Mat> row_imgs;

            for (int c = 0; c < grid_cols; ++c) {
                int idx = r * grid_cols + c;
                if (idx < static_cast<int>(comparisons.size())) {
                    row_imgs.push_back(comparisons[idx]);
                } else {
                    // Add empty placeholder
                    row_imgs.push_back(cv::Mat(comparisons[0].rows, comparisons[0].cols,
                                              CV_8UC3, cv::Scalar(255, 255, 255)));
                }
            }

            cv::Mat row;
            cv::hconcat(row_imgs, row);
            rows.push_back(row);
        }

        cv::Mat grid;
        cv::vconcat(rows, grid);

        return grid;
    }
};

} // namespace camera_aware_depth

#endif // DEPTH_VISUALIZER_H
