#ifndef DEPTH_LOSS_H
#define DEPTH_LOSS_H

#include <torch/torch.h>

namespace camera_aware_depth {

/**
 * @brief Scale-Invariant Logarithmic Loss (SILog)
 *
 * Based on "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network" (Eigen et al., NeurIPS 2014)
 * and "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer" (Ranftl et al., 2020)
 *
 * Formula:
 * L_si = (1/n) Σ(d_i)^2 - λ/(n^2) (Σd_i)^2
 * where d_i = log(y_i) - log(ŷ_i)
 *
 * This loss is invariant to global scale shifts, which is important for monocular depth estimation.
 */
class ScaleInvariantLoss {
public:
    ScaleInvariantLoss(float lambda = 0.5f, float eps = 1e-6f)
        : lambda_(lambda), eps_(eps) {}

    /**
     * @brief Compute scale-invariant loss
     *
     * @param pred_depth Predicted depth (B, 1, H, W)
     * @param gt_depth Ground truth depth (B, 1, H, W)
     * @param valid_mask Optional mask for valid pixels (B, 1, H, W)
     * @return Scalar loss value
     */
    torch::Tensor forward(torch::Tensor pred_depth,
                         torch::Tensor gt_depth,
                         torch::optional<torch::Tensor> valid_mask = torch::nullopt) {

        // Create valid mask if not provided (exclude zeros)
        auto mask = valid_mask.has_value() ?
            valid_mask.value() :
            (gt_depth > eps_);

        // Clamp predictions to avoid log(0)
        pred_depth = torch::clamp(pred_depth, eps_, 1000.0f);
        gt_depth = torch::clamp(gt_depth, eps_, 1000.0f);

        // Compute log difference
        auto log_diff = torch::log(pred_depth) - torch::log(gt_depth);

        // Apply mask
        auto masked_diff = log_diff.masked_select(mask);

        int64_t n = masked_diff.numel();
        if (n == 0) {
            return torch::zeros(1, pred_depth.options());
        }

        // First term: mean of squared log differences
        auto term1 = torch::pow(masked_diff, 2).sum() / n;

        // Second term: squared mean of log differences (penalizes global scale)
        auto term2 = lambda_ * torch::pow(masked_diff.sum(), 2) / (n * n);

        return term1 - term2;
    }

private:
    float lambda_;  // Weight for scale term
    float eps_;     // Small constant to avoid log(0)
};

/**
 * @brief Multi-scale Gradient Matching Loss
 *
 * Based on MiDaS paper (Ranftl et al., 2020)
 * Reference: https://arxiv.org/pdf/1907.01341.pdf (Equation 11)
 *
 * This loss encourages the predicted depth gradients to match the ground truth gradients,
 * which helps preserve edge sharpness and geometric structure.
 *
 * Operates in log-depth space for scale invariance.
 */
class GradientMatchingLoss {
public:
    GradientMatchingLoss(int num_scales = 4, float eps = 1e-6f)
        : num_scales_(num_scales), eps_(eps) {}

    /**
     * @brief Compute multi-scale gradient matching loss
     *
     * @param pred_depth Predicted depth (B, 1, H, W)
     * @param gt_depth Ground truth depth (B, 1, H, W)
     * @param valid_mask Optional mask for valid pixels
     * @return Scalar loss value
     */
    torch::Tensor forward(torch::Tensor pred_depth,
                         torch::Tensor gt_depth,
                         torch::optional<torch::Tensor> valid_mask = torch::nullopt) {

        torch::Tensor total_loss = torch::zeros(1, pred_depth.options());

        for (int scale = 0; scale < num_scales_; ++scale) {
            // Downsample if not at original scale
            auto pred_scale = pred_depth;
            auto gt_scale = gt_depth;

            if (scale > 0) {
                int factor = std::pow(2, scale);
                pred_scale = torch::nn::functional::avg_pool2d(pred_depth,
                    torch::nn::functional::AvgPool2dFuncOptions(factor).stride(factor));
                gt_scale = torch::nn::functional::avg_pool2d(gt_depth,
                    torch::nn::functional::AvgPool2dFuncOptions(factor).stride(factor));
            }

            // Compute log depth
            pred_scale = torch::log(torch::clamp(pred_scale, eps_, 1000.0f));
            gt_scale = torch::log(torch::clamp(gt_scale, eps_, 1000.0f));

            // Compute gradients
            auto loss_scale = computeGradientLoss(pred_scale, gt_scale, valid_mask);
            total_loss = total_loss + loss_scale;
        }

        return total_loss / num_scales_;
    }

private:
    int num_scales_;
    float eps_;

    /**
     * @brief Compute gradient loss at a single scale
     *
     * Uses Sobel-like operators for computing gradients
     */
    torch::Tensor computeGradientLoss(torch::Tensor pred,
                                      torch::Tensor gt,
                                      torch::optional<torch::Tensor> mask) {

        // Compute x-gradients (horizontal)
        auto pred_grad_x = pred.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                      torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}) -
                          pred.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                     torch::indexing::Slice(), torch::indexing::Slice(0, -1)});

        auto gt_grad_x = gt.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                   torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}) -
                        gt.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                 torch::indexing::Slice(), torch::indexing::Slice(0, -1)});

        // Compute y-gradients (vertical)
        auto pred_grad_y = pred.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                      torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}) -
                          pred.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                     torch::indexing::Slice(0, -1), torch::indexing::Slice()});

        auto gt_grad_y = gt.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                   torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}) -
                        gt.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                 torch::indexing::Slice(0, -1), torch::indexing::Slice()});

        // L1 loss on gradients (more robust than L2)
        auto loss_x = torch::abs(pred_grad_x - gt_grad_x).mean();
        auto loss_y = torch::abs(pred_grad_y - gt_grad_y).mean();

        return loss_x + loss_y;
    }
};

/**
 * @brief Edge-Aware Smoothness Loss
 *
 * Encourages smooth depth predictions while preserving edges aligned with image edges.
 * Based on Monodepth (Godard et al., CVPR 2017)
 *
 * Formula:
 * L_smooth = Σ |∇_x d| * exp(-|∇_x I|) + |∇_y d| * exp(-|∇_y I|)
 */
class SmoothnessLoss {
public:
    SmoothnessLoss(float eps = 1e-6f) : eps_(eps) {}

    /**
     * @brief Compute edge-aware smoothness loss
     *
     * @param pred_depth Predicted depth (B, 1, H, W)
     * @param image RGB image (B, 3, H, W) - used to detect edges
     * @return Scalar loss value
     */
    torch::Tensor forward(torch::Tensor pred_depth, torch::Tensor image) {

        // Normalize depth for gradient computation
        auto depth_mean = pred_depth.mean({2, 3}, true);
        auto depth_norm = pred_depth / (depth_mean + eps_);

        // Compute depth gradients
        auto depth_grad_x = torch::abs(
            depth_norm.index({torch::indexing::Slice(), torch::indexing::Slice(),
                            torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}) -
            depth_norm.index({torch::indexing::Slice(), torch::indexing::Slice(),
                            torch::indexing::Slice(), torch::indexing::Slice(0, -1)})
        );

        auto depth_grad_y = torch::abs(
            depth_norm.index({torch::indexing::Slice(), torch::indexing::Slice(),
                            torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}) -
            depth_norm.index({torch::indexing::Slice(), torch::indexing::Slice(),
                            torch::indexing::Slice(0, -1), torch::indexing::Slice()})
        );

        // Compute image gradients (for edge detection)
        auto image_grad_x = torch::abs(
            image.index({torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}) -
            image.index({torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::Slice(), torch::indexing::Slice(0, -1)})
        ).mean(1, true);  // Average over RGB channels

        auto image_grad_y = torch::abs(
            image.index({torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}) -
            image.index({torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::Slice(0, -1), torch::indexing::Slice()})
        ).mean(1, true);

        // Edge-aware weighting: suppress smoothness at image edges
        auto weight_x = torch::exp(-image_grad_x);
        auto weight_y = torch::exp(-image_grad_y);

        // Weighted smoothness loss
        auto loss_x = (depth_grad_x * weight_x).mean();
        auto loss_y = (depth_grad_y * weight_y).mean();

        return loss_x + loss_y;
    }

private:
    float eps_;
};

/**
 * @brief Combined Depth Loss
 *
 * Combines multiple loss terms with configurable weights:
 * - Scale-invariant loss (main depth loss)
 * - Gradient matching loss (preserves structure)
 * - Smoothness loss (encourages smooth predictions)
 */
class CombinedDepthLoss {
public:
    CombinedDepthLoss(float si_weight = 1.0f,
                     float grad_weight = 0.1f,
                     float smooth_weight = 0.001f)
        : si_weight_(si_weight),
          grad_weight_(grad_weight),
          smooth_weight_(smooth_weight),
          si_loss_(),
          grad_loss_(),
          smooth_loss_() {}

    /**
     * @brief Compute combined loss
     *
     * @param pred_depth Predicted depth (B, 1, H, W)
     * @param gt_depth Ground truth depth (B, 1, H, W)
     * @param image RGB image (B, 3, H, W) - for smoothness loss
     * @param valid_mask Optional validity mask
     * @return Scalar loss value
     */
    torch::Tensor forward(torch::Tensor pred_depth,
                         torch::Tensor gt_depth,
                         torch::Tensor image,
                         torch::optional<torch::Tensor> valid_mask = torch::nullopt) {

        auto si = si_loss_.forward(pred_depth, gt_depth, valid_mask);
        auto grad = grad_loss_.forward(pred_depth, gt_depth, valid_mask);
        auto smooth = smooth_loss_.forward(pred_depth, image);

        auto total = si_weight_ * si +
                    grad_weight_ * grad +
                    smooth_weight_ * smooth;

        return total;
    }

    /**
     * @brief Get individual loss components for logging
     */
    std::map<std::string, float> getComponents(torch::Tensor pred_depth,
                                               torch::Tensor gt_depth,
                                               torch::Tensor image,
                                               torch::optional<torch::Tensor> valid_mask = torch::nullopt) {
        std::map<std::string, float> components;

        components["si_loss"] = si_loss_.forward(pred_depth, gt_depth, valid_mask).item<float>();
        components["grad_loss"] = grad_loss_.forward(pred_depth, gt_depth, valid_mask).item<float>();
        components["smooth_loss"] = smooth_loss_.forward(pred_depth, image).item<float>();

        return components;
    }

private:
    float si_weight_;
    float grad_weight_;
    float smooth_weight_;

    ScaleInvariantLoss si_loss_;
    GradientMatchingLoss grad_loss_;
    SmoothnessLoss smooth_loss_;
};

} // namespace camera_aware_depth

#endif // DEPTH_LOSS_H
