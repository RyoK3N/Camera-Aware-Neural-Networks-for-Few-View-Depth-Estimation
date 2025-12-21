#ifndef DEPTH_METRICS_H
#define DEPTH_METRICS_H

#include <torch/torch.h>
#include <map>
#include <string>
#include <vector>
#include <cmath>

namespace camera_aware_depth {

/**
 * @brief Depth Estimation Metrics
 *
 * Standard metrics for monocular depth estimation based on:
 * - Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NeurIPS 2014
 * - Godard et al., "Digging Into Self-Supervised Monocular Depth Estimation", ICCV 2019
 *
 * Common metrics:
 * - AbsRel: Absolute Relative Error
 * - SqRel: Squared Relative Error
 * - RMSE: Root Mean Squared Error
 * - RMSElog: RMSE in log space
 * - δ thresholds: Percentage of pixels where max(pred/gt, gt/pred) < threshold
 *
 * All metrics computed only on valid pixels (depth > 0)
 */
class DepthMetrics {
public:
    /**
     * @brief Compute all depth metrics
     *
     * @param pred_depth Predicted depth (B, 1, H, W) or (B, H, W)
     * @param gt_depth Ground truth depth (B, 1, H, W) or (B, H, W)
     * @param valid_mask Optional validity mask (B, 1, H, W) or (B, H, W)
     * @param min_depth Minimum valid depth (default: 0.1m)
     * @param max_depth Maximum valid depth (default: 10.0m)
     * @return Map of metric name to value
     */
    static std::map<std::string, float> compute(
        torch::Tensor pred_depth,
        torch::Tensor gt_depth,
        torch::optional<torch::Tensor> valid_mask = torch::nullopt,
        float min_depth = 0.1f,
        float max_depth = 10.0f
    ) {
        std::map<std::string, float> metrics;

        // Ensure 4D tensors (B, 1, H, W)
        if (pred_depth.dim() == 3) pred_depth = pred_depth.unsqueeze(1);
        if (gt_depth.dim() == 3) gt_depth = gt_depth.unsqueeze(1);

        // Create validity mask
        auto mask = createValidMask(gt_depth, valid_mask, min_depth, max_depth);

        // Flatten and apply mask
        auto pred_valid = pred_depth.masked_select(mask);
        auto gt_valid = gt_depth.masked_select(mask);

        int64_t num_valid = pred_valid.numel();
        if (num_valid == 0) {
            return getZeroMetrics();
        }

        // Clamp predictions to valid range
        pred_valid = torch::clamp(pred_valid, min_depth, max_depth);

        // Compute metrics
        metrics["abs_rel"] = computeAbsRel(pred_valid, gt_valid);
        metrics["sq_rel"] = computeSqRel(pred_valid, gt_valid);
        metrics["rmse"] = computeRMSE(pred_valid, gt_valid);
        metrics["rmse_log"] = computeRMSElog(pred_valid, gt_valid);
        metrics["mae"] = computeMAE(pred_valid, gt_valid);
        metrics["log10"] = computeLog10(pred_valid, gt_valid);

        // Threshold metrics
        auto delta_metrics = computeDeltaMetrics(pred_valid, gt_valid);
        metrics["delta_1.25"] = delta_metrics[0];
        metrics["delta_1.25^2"] = delta_metrics[1];
        metrics["delta_1.25^3"] = delta_metrics[2];

        // Additional useful metrics
        metrics["num_valid_pixels"] = static_cast<float>(num_valid);
        metrics["mean_pred_depth"] = pred_valid.mean().item<float>();
        metrics["mean_gt_depth"] = gt_valid.mean().item<float>();

        return metrics;
    }

    /**
     * @brief Compute metrics for a batch and return per-sample results
     */
    static std::vector<std::map<std::string, float>> computePerSample(
        torch::Tensor pred_depth,
        torch::Tensor gt_depth,
        torch::optional<torch::Tensor> valid_mask = torch::nullopt,
        float min_depth = 0.1f,
        float max_depth = 10.0f
    ) {
        int batch_size = pred_depth.size(0);
        std::vector<std::map<std::string, float>> results;
        results.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            auto pred_i = pred_depth[i].unsqueeze(0);
            auto gt_i = gt_depth[i].unsqueeze(0);
            torch::optional<torch::Tensor> mask_i = torch::nullopt;

            if (valid_mask.has_value()) {
                mask_i = valid_mask.value()[i].unsqueeze(0);
            }

            results.push_back(compute(pred_i, gt_i, mask_i, min_depth, max_depth));
        }

        return results;
    }

    /**
     * @brief Average metrics across multiple samples
     */
    static std::map<std::string, float> average(
        const std::vector<std::map<std::string, float>>& metrics_list
    ) {
        if (metrics_list.empty()) {
            return getZeroMetrics();
        }

        std::map<std::string, float> avg_metrics;

        // Get all keys from first sample
        for (const auto& [key, _] : metrics_list[0]) {
            float sum = 0.0f;
            for (const auto& metrics : metrics_list) {
                sum += metrics.at(key);
            }
            avg_metrics[key] = sum / metrics_list.size();
        }

        return avg_metrics;
    }

private:
    /**
     * @brief Create validity mask
     */
    static torch::Tensor createValidMask(
        torch::Tensor gt_depth,
        torch::optional<torch::Tensor> user_mask,
        float min_depth,
        float max_depth
    ) {
        // Valid if: depth in [min_depth, max_depth]
        auto mask = (gt_depth > min_depth) & (gt_depth < max_depth);

        // Apply user-provided mask if available
        if (user_mask.has_value()) {
            auto user_m = user_mask.value();
            if (user_m.dim() == 3) user_m = user_m.unsqueeze(1);
            mask = mask & user_m.to(torch::kBool);
        }

        return mask;
    }

    /**
     * @brief Absolute Relative Error: mean(|pred - gt| / gt)
     */
    static float computeAbsRel(torch::Tensor pred, torch::Tensor gt) {
        return (torch::abs(pred - gt) / gt).mean().item<float>();
    }

    /**
     * @brief Squared Relative Error: mean(((pred - gt)^2) / gt)
     */
    static float computeSqRel(torch::Tensor pred, torch::Tensor gt) {
        return (torch::pow(pred - gt, 2) / gt).mean().item<float>();
    }

    /**
     * @brief Root Mean Squared Error: sqrt(mean((pred - gt)^2))
     */
    static float computeRMSE(torch::Tensor pred, torch::Tensor gt) {
        return torch::sqrt(torch::pow(pred - gt, 2).mean()).item<float>();
    }

    /**
     * @brief RMSE in log space: sqrt(mean((log(pred) - log(gt))^2))
     */
    static float computeRMSElog(torch::Tensor pred, torch::Tensor gt) {
        auto log_diff = torch::log(pred) - torch::log(gt);
        return torch::sqrt(torch::pow(log_diff, 2).mean()).item<float>();
    }

    /**
     * @brief Mean Absolute Error: mean(|pred - gt|)
     */
    static float computeMAE(torch::Tensor pred, torch::Tensor gt) {
        return torch::abs(pred - gt).mean().item<float>();
    }

    /**
     * @brief Log10 error: mean(|log10(pred) - log10(gt)|)
     */
    static float computeLog10(torch::Tensor pred, torch::Tensor gt) {
        return torch::abs(torch::log10(pred) - torch::log10(gt)).mean().item<float>();
    }

    /**
     * @brief Delta threshold metrics
     *
     * For each pixel, compute: max(pred/gt, gt/pred)
     * Then compute percentage of pixels where this ratio < threshold
     *
     * Standard thresholds: 1.25, 1.25^2, 1.25^3
     *
     * @return Array of 3 values: [δ<1.25, δ<1.25^2, δ<1.25^3]
     */
    static std::array<float, 3> computeDeltaMetrics(torch::Tensor pred, torch::Tensor gt) {
        // Compute max(pred/gt, gt/pred) for each pixel
        auto ratio = torch::max(pred / gt, gt / pred);

        std::array<float, 3> deltas;
        std::array<float, 3> thresholds = {1.25f, 1.25f * 1.25f, 1.25f * 1.25f * 1.25f};

        for (int i = 0; i < 3; ++i) {
            // Percentage of pixels below threshold
            auto below_threshold = (ratio < thresholds[i]).to(torch::kFloat32);
            deltas[i] = below_threshold.mean().item<float>();
        }

        return deltas;
    }

    /**
     * @brief Get zero metrics (for edge cases)
     */
    static std::map<std::string, float> getZeroMetrics() {
        return {
            {"abs_rel", 0.0f},
            {"sq_rel", 0.0f},
            {"rmse", 0.0f},
            {"rmse_log", 0.0f},
            {"mae", 0.0f},
            {"log10", 0.0f},
            {"delta_1.25", 0.0f},
            {"delta_1.25^2", 0.0f},
            {"delta_1.25^3", 0.0f},
            {"num_valid_pixels", 0.0f},
            {"mean_pred_depth", 0.0f},
            {"mean_gt_depth", 0.0f}
        };
    }
};

/**
 * @brief Metrics Accumulator for tracking metrics across batches
 */
class MetricsAccumulator {
public:
    MetricsAccumulator() : count_(0) {}

    /**
     * @brief Update accumulator with new metrics
     */
    void update(const std::map<std::string, float>& metrics) {
        for (const auto& [key, value] : metrics) {
            running_sum_[key] += value;
        }
        count_++;
    }

    /**
     * @brief Get average metrics
     */
    std::map<std::string, float> average() const {
        if (count_ == 0) {
            return {};
        }

        std::map<std::string, float> avg_metrics;
        for (const auto& [key, sum] : running_sum_) {
            avg_metrics[key] = sum / count_;
        }
        return avg_metrics;
    }

    /**
     * @brief Reset accumulator
     */
    void reset() {
        running_sum_.clear();
        count_ = 0;
    }

    /**
     * @brief Get sample count
     */
    int64_t count() const { return count_; }

private:
    std::map<std::string, float> running_sum_;
    int64_t count_;
};

/**
 * @brief Pretty print metrics
 */
inline std::string formatMetrics(const std::map<std::string, float>& metrics) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);

    // Error metrics (lower is better)
    ss << "Error Metrics:\n";
    ss << "  AbsRel:  " << metrics.at("abs_rel") << "\n";
    ss << "  RMSE:    " << metrics.at("rmse") << "\n";
    ss << "  RMSElog: " << metrics.at("rmse_log") << "\n";
    ss << "  MAE:     " << metrics.at("mae") << "\n";

    // Accuracy metrics (higher is better)
    ss << "\nAccuracy Metrics (%):\n";
    ss << "  δ < 1.25:    " << (metrics.at("delta_1.25") * 100.0f) << "%\n";
    ss << "  δ < 1.25²:   " << (metrics.at("delta_1.25^2") * 100.0f) << "%\n";
    ss << "  δ < 1.25³:   " << (metrics.at("delta_1.25^3") * 100.0f) << "%\n";

    // Statistics
    ss << "\nStatistics:\n";
    ss << "  Valid pixels: " << static_cast<int>(metrics.at("num_valid_pixels")) << "\n";
    ss << "  Mean pred:    " << metrics.at("mean_pred_depth") << "m\n";
    ss << "  Mean GT:      " << metrics.at("mean_gt_depth") << "m\n";

    return ss.str();
}

} // namespace camera_aware_depth

#endif // DEPTH_METRICS_H
