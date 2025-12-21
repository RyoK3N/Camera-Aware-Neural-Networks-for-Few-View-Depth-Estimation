#ifndef PCL_LAYER_H
#define PCL_LAYER_H

#include <torch/torch.h>
#include <cmath>

namespace camera_aware_depth {

/**
 * @brief Perspective Correction Layer (PCL)
 *
 * Based on Spatial Transformer Networks (Jaderberg et al., NIPS 2015)
 * Paper: https://arxiv.org/abs/1506.02025
 *
 * Adapted for depth estimation with 3D geometric transformations using camera intrinsics.
 * The PCL learns to apply perspective-aware transformations to feature maps, allowing
 * the network to reason about 3D geometry and camera perspective.
 *
 * Key differences from standard STN:
 * - Conditioned on camera intrinsics (focal length, principal point)
 * - Uses 3D ray directions for geometric reasoning
 * - Applies perspective-aware warping instead of generic affine transforms
 *
 * References:
 * - Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
 * - Perspective Transformer Nets: https://arxiv.org/abs/1612.00814
 * - DeepV2D: https://arxiv.org/abs/1812.04605 (3D feature transformation)
 */
struct PerspectiveCorrectionLayerImpl : torch::nn::Module {
    torch::nn::Linear localization_fc1{nullptr};
    torch::nn::Linear localization_fc2{nullptr};
    torch::nn::Linear fc_transform{nullptr};

    int feature_channels_;
    int hidden_dim_;

    /**
     * @brief Construct Perspective Correction Layer
     *
     * @param feature_channels Number of input feature channels
     * @param camera_dim Dimension of camera parameter vector (typically 4: fx, fy, cx, cy)
     * @param hidden_dim Hidden dimension for localization network
     */
    PerspectiveCorrectionLayerImpl(int feature_channels,
                                   int camera_dim = 4,
                                   int hidden_dim = 128)
        : feature_channels_(feature_channels),
          hidden_dim_(hidden_dim) {

        // Localization network: predicts transformation parameters
        // Input: concatenated global pooled features + camera intrinsics
        localization_fc1 = register_module("loc_fc1",
            torch::nn::Linear(feature_channels + camera_dim, hidden_dim));

        localization_fc2 = register_module("loc_fc2",
            torch::nn::Linear(hidden_dim, hidden_dim));

        // Output: 6 parameters for 2D affine transformation
        // [scale_x, scale_y, translate_x, translate_y, rotation, shear]
        fc_transform = register_module("fc_transform",
            torch::nn::Linear(hidden_dim, 6));

        // Initialize transform to identity
        torch::nn::init::zeros_(fc_transform->weight);
        fc_transform->bias.data() = torch::tensor({1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    }

    /**
     * @brief Forward pass
     *
     * @param features Input feature map (B, C, H, W)
     * @param camera_intrinsics Camera intrinsics (B, 4) - [fx, fy, cx, cy]
     * @param ray_directions Optional ray directions (B, 3, H, W) for 3D-aware transformation
     * @return Perspective-corrected features (B, C, H, W)
     */
    torch::Tensor forward(torch::Tensor features,
                         torch::Tensor camera_intrinsics,
                         torch::optional<torch::Tensor> ray_directions = torch::nullopt) {

        int batch_size = features.size(0);
        int height = features.size(2);
        int width = features.size(3);

        // Global average pooling to get feature summary
        auto pooled = torch::adaptive_avg_pool2d(features, {1, 1});
        pooled = pooled.view({batch_size, -1});  // (B, C)

        // Concatenate with camera intrinsics
        auto loc_input = torch::cat({pooled, camera_intrinsics}, 1);  // (B, C + 4)

        // Localization network
        auto h = torch::relu(localization_fc1(loc_input));
        h = torch::relu(localization_fc2(h));
        auto transform_params = fc_transform(h);  // (B, 6)

        // Build affine transformation matrix from parameters
        auto theta = buildAffineMatrix(transform_params);  // (B, 2, 3)

        // Generate sampling grid
        // Note: Using direct call as AffineGridFuncOptions may not be available in all LibTorch versions
        auto grid = torch::affine_grid_generator(theta, features.sizes(), /*align_corners=*/false);

        // Apply transformation using grid sampling
        auto transformed = torch::nn::functional::grid_sample(features, grid,
            torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false));

        return transformed;
    }

    /**
     * @brief Forward with 3D-aware perspective correction
     *
     * Uses ray directions to perform perspective-aware transformation in 3D space
     */
    torch::Tensor forward3D(torch::Tensor features,
                           torch::Tensor camera_intrinsics,
                           torch::Tensor ray_directions) {

        // First apply standard perspective correction
        auto corrected = forward(features, camera_intrinsics);

        // Apply ray-direction-based weighting for 3D consistency
        // Weight features based on viewing angle (z-component of ray direction)
        auto ray_z = ray_directions.index({torch::indexing::Slice(),
                                          2,
                                          torch::indexing::Slice(),
                                          torch::indexing::Slice()}).unsqueeze(1);

        // Normalize to [0, 1] and use as attention weight
        // Areas viewed from steep angles get less weight
        auto weight = torch::abs(ray_z);  // Closer to 1 = more perpendicular view

        // Apply soft gating
        corrected = corrected * (0.5f + 0.5f * weight);

        return corrected;
    }

private:
    /**
     * @brief Build 2x3 affine transformation matrix from parameters
     *
     * Parameters: [scale_x, scale_y, translate_x, translate_y, rotation, shear]
     *
     * The affine matrix has the form:
     * [[scale_x * cos(θ), -sin(θ) + shear, translate_x],
     *  [sin(θ), scale_y * cos(θ), translate_y]]
     */
    torch::Tensor buildAffineMatrix(torch::Tensor params) {
        int batch_size = params.size(0);

        auto scale_x = params.index({torch::indexing::Slice(), 0});
        auto scale_y = params.index({torch::indexing::Slice(), 1});
        auto trans_x = params.index({torch::indexing::Slice(), 2});
        auto trans_y = params.index({torch::indexing::Slice(), 3});
        auto rotation = params.index({torch::indexing::Slice(), 4});
        auto shear = params.index({torch::indexing::Slice(), 5});

        auto cos_r = torch::cos(rotation);
        auto sin_r = torch::sin(rotation);

        // Build affine matrix
        auto theta = torch::zeros({batch_size, 2, 3}, params.options());

        // First row: [scale_x * cos(θ), -sin(θ) + shear, translate_x]
        theta.index_put_({torch::indexing::Slice(), 0, 0}, scale_x * cos_r);
        theta.index_put_({torch::indexing::Slice(), 0, 1}, -sin_r + shear);
        theta.index_put_({torch::indexing::Slice(), 0, 2}, trans_x);

        // Second row: [sin(θ), scale_y * cos(θ), translate_y]
        theta.index_put_({torch::indexing::Slice(), 1, 0}, sin_r);
        theta.index_put_({torch::indexing::Slice(), 1, 1}, scale_y * cos_r);
        theta.index_put_({torch::indexing::Slice(), 1, 2}, trans_y);

        return theta;
    }
};
TORCH_MODULE(PerspectiveCorrectionLayer);

/**
 * @brief 3D Perspective Transformer
 *
 * More advanced version that explicitly models 3D geometry using depth and ray directions.
 * Based on "Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction
 * without 3D Supervision" (Yan et al., NeurIPS 2016)
 * Paper: https://arxiv.org/abs/1612.00814
 *
 * This layer learns to warp features in 3D space using predicted depth and camera geometry.
 */
struct Perspective3DTransformerImpl : torch::nn::Module {
    torch::nn::Conv2d depth_predictor{nullptr};
    torch::nn::Linear localization_fc1{nullptr};
    torch::nn::Linear localization_fc2{nullptr};

    int feature_channels_;

    Perspective3DTransformerImpl(int feature_channels, int camera_dim = 4)
        : feature_channels_(feature_channels) {

        // Predict per-pixel depth offsets for warping
        depth_predictor = register_module("depth_pred",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_channels, 1, 1)));

        // Localization network for global transformation
        localization_fc1 = register_module("loc_fc1",
            torch::nn::Linear(feature_channels + camera_dim, 128));

        localization_fc2 = register_module("loc_fc2",
            torch::nn::Linear(128, 3));  // [tx, ty, tz] translation

        // Initialize to no translation
        torch::nn::init::zeros_(localization_fc2->weight);
        torch::nn::init::zeros_(localization_fc2->bias);
    }

    /**
     * @brief Forward pass with 3D perspective transformation
     *
     * @param features Input feature map (B, C, H, W)
     * @param camera_intrinsics Camera intrinsics (B, 4) - [fx, fy, cx, cy]
     * @param ray_directions Ray directions (B, 3, H, W)
     * @return 3D perspective-corrected features (B, C, H, W)
     */
    torch::Tensor forward(torch::Tensor features,
                         torch::Tensor camera_intrinsics,
                         torch::Tensor ray_directions) {

        int batch_size = features.size(0);
        int height = features.size(2);
        int width = features.size(3);

        // Predict depth offsets
        auto depth_offset = torch::tanh(depth_predictor(features));  // (B, 1, H, W), range [-1, 1]

        // Global transformation parameters
        auto pooled = torch::adaptive_avg_pool2d(features, {1, 1}).view({batch_size, -1});
        auto loc_input = torch::cat({pooled, camera_intrinsics}, 1);
        auto h = torch::relu(localization_fc1(loc_input));
        auto translation = localization_fc2(h);  // (B, 3)

        // Compute 3D positions using ray directions and depth offsets
        // P_3D = depth * ray_direction + translation
        auto tx = translation.index({torch::indexing::Slice(), 0}).view({batch_size, 1, 1, 1});
        auto ty = translation.index({torch::indexing::Slice(), 1}).view({batch_size, 1, 1, 1});
        auto tz = translation.index({torch::indexing::Slice(), 2}).view({batch_size, 1, 1, 1});

        // Apply depth-based displacement along ray directions
        auto dx = depth_offset * ray_directions.index({torch::indexing::Slice(), 0,
                                                       torch::indexing::Slice(),
                                                       torch::indexing::Slice()}).unsqueeze(1);
        auto dy = depth_offset * ray_directions.index({torch::indexing::Slice(), 1,
                                                       torch::indexing::Slice(),
                                                       torch::indexing::Slice()}).unsqueeze(1);

        // Add global translation
        dx = dx + tx;
        dy = dy + ty;

        // Create normalized sampling grid [-1, 1]
        auto grid_x = dx / (width / 2.0f);
        auto grid_y = dy / (height / 2.0f);

        // Stack to create grid (B, H, W, 2)
        auto grid = torch::stack({grid_x.squeeze(1), grid_y.squeeze(1)}, -1);

        // Apply grid sampling
        auto transformed = torch::nn::functional::grid_sample(features, grid,
            torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder)
                .align_corners(true));

        return transformed;
    }
};
TORCH_MODULE(Perspective3DTransformer);

} // namespace camera_aware_depth

#endif // PCL_LAYER_H
