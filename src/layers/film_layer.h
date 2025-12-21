#ifndef FILM_LAYER_H
#define FILM_LAYER_H

#include <torch/torch.h>

namespace camera_aware_depth {

/**
 * @brief Feature-wise Linear Modulation (FiLM) Layer
 *
 * Based on "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., AAAI 2018)
 * Paper: https://arxiv.org/abs/1709.07871
 * Interactive explanation: https://distill.pub/2018/feature-wise-transformations/
 *
 * FiLM performs feature-wise affine transformation conditioned on external input:
 * FiLM(F; γ, β) = γ ⊙ F + β
 *
 * where:
 * - F: input feature map (B, C, H, W)
 * - γ: scale parameters (B, C)
 * - β: shift parameters (B, C)
 * - ⊙: element-wise multiplication (broadcasted)
 *
 * This allows the network to adapt its features based on camera parameters.
 */
struct FiLMLayerImpl : torch::nn::Module {
    // Camera parameter embedding network
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc_gamma{nullptr};  // Outputs scale parameters
    torch::nn::Linear fc_beta{nullptr};   // Outputs shift parameters

    int feature_channels_;
    bool use_batch_norm_;

    torch::nn::BatchNorm1d bn1{nullptr};
    torch::nn::BatchNorm1d bn2{nullptr};

    /**
     * @brief Construct FiLM layer
     *
     * @param camera_dim Dimension of camera parameter vector
     * @param feature_channels Number of feature channels to modulate
     * @param hidden_dim Hidden dimension for embedding network
     * @param use_batch_norm Whether to use batch normalization
     */
    FiLMLayerImpl(int camera_dim,
                  int feature_channels,
                  int hidden_dim = 256,
                  bool use_batch_norm = true)
        : feature_channels_(feature_channels),
          use_batch_norm_(use_batch_norm) {

        // Camera embedding network (MLP)
        fc1 = register_module("fc1", torch::nn::Linear(camera_dim, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, hidden_dim));

        // Separate heads for gamma and beta
        fc_gamma = register_module("fc_gamma", torch::nn::Linear(hidden_dim, feature_channels));
        fc_beta = register_module("fc_beta", torch::nn::Linear(hidden_dim, feature_channels));

        if (use_batch_norm_) {
            bn1 = register_module("bn1", torch::nn::BatchNorm1d(128));
            bn2 = register_module("bn2", torch::nn::BatchNorm1d(hidden_dim));
        }

        // Initialize gamma close to 1 and beta close to 0
        torch::nn::init::normal_(fc_gamma->weight, 0.0, 0.01);
        torch::nn::init::constant_(fc_gamma->bias, 1.0);  // Start with identity transform
        torch::nn::init::normal_(fc_beta->weight, 0.0, 0.01);
        torch::nn::init::constant_(fc_beta->bias, 0.0);
    }

    /**
     * @brief Forward pass
     *
     * @param features Input feature map (B, C, H, W)
     * @param camera_params Camera parameters (B, camera_dim)
     *                     Can be intrinsics only, or concat of intrinsics + extrinsics
     * @return Modulated features (B, C, H, W)
     */
    torch::Tensor forward(torch::Tensor features, torch::Tensor camera_params) {
        // Embed camera parameters through MLP
        auto h = fc1(camera_params);
        if (use_batch_norm_ && h.size(0) > 1) {  // BatchNorm requires batch size > 1
            h = bn1(h);
        }
        h = torch::relu(h);

        h = fc2(h);
        if (use_batch_norm_ && h.size(0) > 1) {
            h = bn2(h);
        }
        h = torch::relu(h);

        // Generate scale (gamma) and shift (beta) parameters
        auto gamma = fc_gamma(h);  // (B, C)
        auto beta = fc_beta(h);    // (B, C)

        // Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1);
        beta = beta.unsqueeze(-1).unsqueeze(-1);

        // Apply FiLM transformation
        auto modulated = gamma * features + beta;

        return modulated;
    }

    /**
     * @brief Get gamma and beta for analysis
     *
     * Useful for visualizing how camera parameters affect features
     */
    std::pair<torch::Tensor, torch::Tensor> get_modulation_params(torch::Tensor camera_params) {
        auto h = torch::relu(fc1(camera_params));
        if (use_batch_norm_ && h.size(0) > 1) {
            h = bn1(h);
        }

        h = torch::relu(fc2(h));
        if (use_batch_norm_ && h.size(0) > 1) {
            h = bn2(h);
        }

        auto gamma = fc_gamma(h);
        auto beta = fc_beta(h);

        return std::make_pair(gamma, beta);
    }
};
TORCH_MODULE(FiLMLayer);

/**
 * @brief FiLM-conditioned Convolution Block
 *
 * Combines convolutional layers with FiLM conditioning.
 * Useful for building camera-aware encoder/decoder blocks.
 */
struct FiLMConvBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    FiLMLayer film{nullptr};

    FiLMConvBlockImpl(int in_channels,
                     int out_channels,
                     int camera_dim,
                     int kernel_size = 3) {

        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .padding(kernel_size / 2)
                .bias(false)
        ));

        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));

        film = register_module("film", FiLMLayer(camera_dim, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_params) {
        x = conv(x);
        x = bn(x);
        x = film(x, camera_params);
        x = torch::relu(x);
        return x;
    }
};
TORCH_MODULE(FiLMConvBlock);

} // namespace camera_aware_depth

#endif // FILM_LAYER_H
