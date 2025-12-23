#ifndef DEPTH_VIZ_H
#define DEPTH_VIZ_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace camera_aware_depth {

/**
 * @brief Utilities for visualizing depth predictions
 */
class DepthVisualizer {
public:
    /**
     * @brief Create a visualization comparing prediction vs ground truth
     *
     * @param pred Predicted depth (1, H, W) or (H, W)
     * @param gt Ground truth depth (1, H, W) or (H, W)
     * @param rgb Optional RGB image (3, H, W)
     * @return Visualization tensor (3, H, W_combined)
     */
    static torch::Tensor createComparisonViz(
        const torch::Tensor& pred,
        const torch::Tensor& gt,
        const torch::Tensor& rgb = torch::Tensor()
    ) {
        auto pred_2d = pred.dim() == 3 ? pred.squeeze(0) : pred;
        auto gt_2d = gt.dim() == 3 ? gt.squeeze(0) : gt;

        // Normalize to [0, 1]
        auto pred_norm = normalizeDepth(pred_2d);
        auto gt_norm = normalizeDepth(gt_2d);

        // Create error map
        auto error = torch::abs(pred_2d - gt_2d);
        auto error_norm = normalizeDepth(error);

        // Convert to color
        auto pred_color = applyColormap(pred_norm);
        auto gt_color = applyColormap(gt_norm);
        auto error_color = applyColormap(error_norm, "hot");

        // Concatenate horizontally
        std::vector<torch::Tensor> viz_list;

        if (rgb.defined() && rgb.numel() > 0) {
            viz_list.push_back(rgb);
        }

        viz_list.push_back(gt_color);
        viz_list.push_back(pred_color);
        viz_list.push_back(error_color);

        return torch::cat(viz_list, /*dim=*/2);  // Concatenate along width
    }

    /**
     * @brief Save depth map as colored image
     */
    static void saveDepthImage(
        const torch::Tensor& depth,
        const std::string& filename,
        const std::string& colormap = "viridis"
    ) {
        auto depth_2d = depth.dim() == 3 ? depth.squeeze(0) : depth;
        auto depth_norm = normalizeDepth(depth_2d);
        auto depth_color = applyColormap(depth_norm, colormap);

        // Convert to cv::Mat
        auto depth_cpu = depth_color.cpu().permute({1, 2, 0}).contiguous();
        int H = depth_cpu.size(0);
        int W = depth_cpu.size(1);

        cv::Mat img(H, W, CV_32FC3, depth_cpu.data_ptr<float>());
        img.convertTo(img, CV_8UC3, 255.0);

        // Convert RGB to BGR for OpenCV
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, img);
    }

private:
    /**
     * @brief Normalize depth to [0, 1] range
     */
    static torch::Tensor normalizeDepth(const torch::Tensor& depth) {
        auto valid_mask = depth > 0;
        auto valid_depth = depth.masked_select(valid_mask);

        if (valid_depth.numel() == 0) {
            return depth;
        }

        float min_val = valid_depth.min().item<float>();
        float max_val = valid_depth.max().item<float>();

        if (max_val - min_val < 1e-6) {
            return torch::zeros_like(depth);
        }

        auto normalized = (depth - min_val) / (max_val - min_val);
        normalized = torch::where(valid_mask, normalized, torch::zeros_like(depth));

        return torch::clamp(normalized, 0.0f, 1.0f);
    }

    /**
     * @brief Apply colormap to grayscale tensor
     */
    static torch::Tensor applyColormap(
        const torch::Tensor& gray,
        const std::string& colormap = "viridis"
    ) {
        auto gray_cpu = gray.cpu();
        int H = gray_cpu.size(0);
        int W = gray_cpu.size(1);

        // Convert to cv::Mat
        cv::Mat gray_mat(H, W, CV_32F, gray_cpu.data_ptr<float>());
        gray_mat.convertTo(gray_mat, CV_8U, 255.0);

        // Apply colormap
        cv::Mat colored;
        if (colormap == "hot") {
            cv::applyColorMap(gray_mat, colored, cv::COLORMAP_HOT);
        } else if (colormap == "jet") {
            cv::applyColorMap(gray_mat, colored, cv::COLORMAP_JET);
        } else {  // Default: viridis
            cv::applyColorMap(gray_mat, colored, cv::COLORMAP_VIRIDIS);
        }

        // Convert back to tensor
        colored.convertTo(colored, CV_32FC3, 1.0 / 255.0);

        // BGR to RGB
        cv::cvtColor(colored, colored, cv::COLOR_BGR2RGB);

        // Mat to tensor (H, W, 3) -> (3, H, W)
        auto tensor = torch::from_blob(
            colored.data,
            {H, W, 3},
            torch::kFloat32
        ).clone();

        return tensor.permute({2, 0, 1});
    }
};

} // namespace camera_aware_depth

#endif // DEPTH_VIZ_H
