#ifndef INTRINSICS_UNET_H
#define INTRINSICS_UNET_H

#include <torch/torch.h>
#include "../layers/film_layer.h"
#include "../layers/spatial_attention.h"

namespace camera_aware_depth {

/**
 * @brief FiLM-Conditioned Double Convolution Block
 *
 * U-Net building block with FiLM conditioning for camera intrinsics.
 * Combines: Conv -> BatchNorm -> ReLU -> FiLM -> Conv -> BatchNorm -> ReLU
 */
struct FiLMDoubleConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    FiLMLayer film{nullptr};

    FiLMDoubleConvImpl(int in_channels, int out_channels, int camera_dim = 4) {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).bias(false)
        ));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(false)
        ));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

        // FiLM layer for camera conditioning
        film = register_module("film", FiLMLayer(camera_dim, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_params) {
        x = conv1(x);
        x = bn1(x);
        x = torch::relu(x);

        // Apply FiLM conditioning
        x = film(x, camera_params);

        x = conv2(x);
        x = bn2(x);
        x = torch::relu(x);

        return x;
    }
};
TORCH_MODULE(FiLMDoubleConv);

/**
 * @brief FiLM-Conditioned Encoder Block
 *
 * MaxPool -> FiLMDoubleConv
 */
struct FiLMEncoderBlockImpl : torch::nn::Module {
    torch::nn::MaxPool2d pool{nullptr};
    FiLMDoubleConv conv{nullptr};

    FiLMEncoderBlockImpl(int in_channels, int out_channels, int camera_dim = 4) {
        pool = register_module("pool", torch::nn::MaxPool2d(2));
        conv = register_module("conv", FiLMDoubleConv(in_channels, out_channels, camera_dim));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_params) {
        x = pool(x);
        x = conv(x, camera_params);
        return x;
    }
};
TORCH_MODULE(FiLMEncoderBlock);

/**
 * @brief FiLM-Conditioned Decoder Block
 *
 * UpConv -> Concatenate with skip -> FiLMDoubleConv
 */
struct FiLMDecoderBlockImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d up{nullptr};
    FiLMDoubleConv conv{nullptr};

    FiLMDecoderBlockImpl(int in_channels, int out_channels, int camera_dim = 4) {
        up = register_module("up", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 2).stride(2)
        ));
        conv = register_module("conv", FiLMDoubleConv(in_channels, out_channels, camera_dim));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor skip, torch::Tensor camera_params) {
        x = up(x);

        // Handle size mismatch by padding
        int diff_h = skip.size(2) - x.size(2);
        int diff_w = skip.size(3) - x.size(3);

        if (diff_h > 0 || diff_w > 0) {
            x = torch::nn::functional::pad(x,
                torch::nn::functional::PadFuncOptions({diff_w / 2, diff_w - diff_w / 2,
                                                       diff_h / 2, diff_h - diff_h / 2}));
        }

        // Concatenate along channel dimension
        x = torch::cat({skip, x}, 1);
        x = conv(x, camera_params);

        return x;
    }
};
TORCH_MODULE(FiLMDecoderBlock);

/**
 * @brief Intrinsics-Conditioned U-Net
 *
 * U-Net architecture conditioned on camera intrinsics using FiLM layers.
 * Based on:
 * - U-Net: Ronneberger et al., MICCAI 2015
 * - FiLM: Perez et al., AAAI 2018
 *
 * This network adapts its features based on camera parameters, allowing it to
 * handle varying camera intrinsics (focal length, principal point, etc.)
 *
 * Architecture:
 * - Input: RGB image (B, 3, H, W) + Camera intrinsics (B, 4)
 * - Encoder: 4 FiLM-conditioned downsampling blocks
 * - Bottleneck: FiLM-conditioned features
 * - Decoder: 4 FiLM-conditioned upsampling blocks with skip connections
 * - Output: Depth map (B, 1, H, W)
 *
 * Camera intrinsics format: [fx, fy, cx, cy]
 * - fx, fy: focal lengths in pixels
 * - cx, cy: principal point coordinates
 */
struct IntrinsicsConditionedUNetImpl : torch::nn::Module {
    // Encoder
    FiLMDoubleConv enc1{nullptr};
    FiLMEncoderBlock enc2{nullptr};
    FiLMEncoderBlock enc3{nullptr};
    FiLMEncoderBlock enc4{nullptr};

    // Bottleneck
    FiLMEncoderBlock bottleneck{nullptr};

    // Decoder
    FiLMDecoderBlock dec4{nullptr};
    FiLMDecoderBlock dec3{nullptr};
    FiLMDecoderBlock dec2{nullptr};
    FiLMDecoderBlock dec1{nullptr};

    // Output
    torch::nn::Conv2d out_conv{nullptr};

    // Configuration
    float max_depth;
    int camera_dim_;

    /**
     * @brief Construct Intrinsics-Conditioned U-Net
     *
     * @param in_channels Number of input channels (default: 3 for RGB)
     * @param init_features Initial number of features (default: 64)
     * @param camera_dim Dimension of camera parameter vector (default: 4 for [fx, fy, cx, cy])
     * @param max_depth_value Maximum depth value for output scaling (default: 10.0)
     */
    IntrinsicsConditionedUNetImpl(int in_channels = 3,
                                  int init_features = 64,
                                  int camera_dim = 4,
                                  float max_depth_value = 10.0f)
        : max_depth(max_depth_value), camera_dim_(camera_dim) {

        // Encoder pathway
        enc1 = register_module("enc1", FiLMDoubleConv(in_channels, init_features, camera_dim));
        enc2 = register_module("enc2", FiLMEncoderBlock(init_features, init_features * 2, camera_dim));
        enc3 = register_module("enc3", FiLMEncoderBlock(init_features * 2, init_features * 4, camera_dim));
        enc4 = register_module("enc4", FiLMEncoderBlock(init_features * 4, init_features * 8, camera_dim));

        // Bottleneck
        bottleneck = register_module("bottleneck",
            FiLMEncoderBlock(init_features * 8, init_features * 16, camera_dim));

        // Decoder pathway
        dec4 = register_module("dec4", FiLMDecoderBlock(init_features * 16, init_features * 8, camera_dim));
        dec3 = register_module("dec3", FiLMDecoderBlock(init_features * 8, init_features * 4, camera_dim));
        dec2 = register_module("dec2", FiLMDecoderBlock(init_features * 4, init_features * 2, camera_dim));
        dec1 = register_module("dec1", FiLMDecoderBlock(init_features * 2, init_features, camera_dim));

        // Output layer: 1x1 convolution to get depth map
        out_conv = register_module("out_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(init_features, 1, 1)
        ));
    }

    /**
     * @brief Forward pass
     *
     * @param x Input RGB image (B, 3, H, W)
     * @param camera_intrinsics Camera intrinsics (B, 4) - [fx, fy, cx, cy]
     * @return Depth map (B, 1, H, W) in range [0, max_depth]
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_intrinsics) {
        // Normalize camera intrinsics for better conditioning
        // Divide focal lengths by image width/height for scale invariance
        auto normalized_intrinsics = normalizeCameraIntrinsics(camera_intrinsics, x.size(3), x.size(2));

        // Encoder with skip connections
        auto skip1 = enc1(x, normalized_intrinsics);
        auto skip2 = enc2(skip1, normalized_intrinsics);
        auto skip3 = enc3(skip2, normalized_intrinsics);
        auto skip4 = enc4(skip3, normalized_intrinsics);

        // Bottleneck
        auto x_bottleneck = bottleneck(skip4, normalized_intrinsics);

        // Decoder with skip connections
        x = dec4(x_bottleneck, skip4, normalized_intrinsics);
        x = dec3(x, skip3, normalized_intrinsics);
        x = dec2(x, skip2, normalized_intrinsics);
        x = dec1(x, skip1, normalized_intrinsics);

        // Output: sigmoid activation to get [0, 1], then scale to [0, max_depth]
        x = out_conv(x);
        x = torch::sigmoid(x) * max_depth;

        return x;
    }

    /**
     * @brief Get number of parameters
     */
    int64_t count_parameters() {
        int64_t total = 0;
        for (const auto& param : this->parameters()) {
            total += param.numel();
        }
        return total;
    }

private:
    /**
     * @brief Normalize camera intrinsics for better conditioning
     *
     * Normalizes focal lengths by image dimensions to make them scale-invariant
     *
     * @param intrinsics Camera intrinsics (B, 4) - [fx, fy, cx, cy]
     * @param width Image width
     * @param height Image height
     * @return Normalized intrinsics (B, 4)
     */
    torch::Tensor normalizeCameraIntrinsics(torch::Tensor intrinsics, int width, int height) {
        auto normalized = intrinsics.clone();

        // Normalize focal lengths by image dimensions
        normalized.index_put_({torch::indexing::Slice(), 0},
            intrinsics.index({torch::indexing::Slice(), 0}) / width);   // fx / width
        normalized.index_put_({torch::indexing::Slice(), 1},
            intrinsics.index({torch::indexing::Slice(), 1}) / height);  // fy / height

        // Normalize principal point to [-1, 1]
        normalized.index_put_({torch::indexing::Slice(), 2},
            (intrinsics.index({torch::indexing::Slice(), 2}) / width) * 2.0f - 1.0f);   // cx
        normalized.index_put_({torch::indexing::Slice(), 3},
            (intrinsics.index({torch::indexing::Slice(), 3}) / height) * 2.0f - 1.0f);  // cy

        return normalized;
    }
};
TORCH_MODULE(IntrinsicsConditionedUNet);

/**
 * @brief Intrinsics-Conditioned U-Net with Attention
 *
 * Enhanced version with CBAM attention modules in the decoder.
 * Attention helps the network focus on important regions and depth discontinuities.
 */
struct IntrinsicsAttentionUNetImpl : torch::nn::Module {
    // Encoder
    FiLMDoubleConv enc1{nullptr};
    FiLMEncoderBlock enc2{nullptr};
    FiLMEncoderBlock enc3{nullptr};
    FiLMEncoderBlock enc4{nullptr};

    // Bottleneck
    FiLMEncoderBlock bottleneck{nullptr};

    // Decoder
    FiLMDecoderBlock dec4{nullptr};
    FiLMDecoderBlock dec3{nullptr};
    FiLMDecoderBlock dec2{nullptr};
    FiLMDecoderBlock dec1{nullptr};

    // Attention modules
    CBAM att4{nullptr};
    CBAM att3{nullptr};
    CBAM att2{nullptr};
    CBAM att1{nullptr};

    // Output
    torch::nn::Conv2d out_conv{nullptr};

    float max_depth;

    IntrinsicsAttentionUNetImpl(int in_channels = 3,
                               int init_features = 64,
                               int camera_dim = 4,
                               float max_depth_value = 10.0f)
        : max_depth(max_depth_value) {

        // Encoder
        enc1 = register_module("enc1", FiLMDoubleConv(in_channels, init_features, camera_dim));
        enc2 = register_module("enc2", FiLMEncoderBlock(init_features, init_features * 2, camera_dim));
        enc3 = register_module("enc3", FiLMEncoderBlock(init_features * 2, init_features * 4, camera_dim));
        enc4 = register_module("enc4", FiLMEncoderBlock(init_features * 4, init_features * 8, camera_dim));

        // Bottleneck
        bottleneck = register_module("bottleneck",
            FiLMEncoderBlock(init_features * 8, init_features * 16, camera_dim));

        // Decoder
        dec4 = register_module("dec4", FiLMDecoderBlock(init_features * 16, init_features * 8, camera_dim));
        dec3 = register_module("dec3", FiLMDecoderBlock(init_features * 8, init_features * 4, camera_dim));
        dec2 = register_module("dec2", FiLMDecoderBlock(init_features * 4, init_features * 2, camera_dim));
        dec1 = register_module("dec1", FiLMDecoderBlock(init_features * 2, init_features, camera_dim));

        // Attention modules in decoder
        att4 = register_module("att4", CBAM(init_features * 8));
        att3 = register_module("att3", CBAM(init_features * 4));
        att2 = register_module("att2", CBAM(init_features * 2));
        att1 = register_module("att1", CBAM(init_features));

        // Output
        out_conv = register_module("out_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(init_features, 1, 1)
        ));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_intrinsics) {
        auto normalized_intrinsics = normalizeCameraIntrinsics(camera_intrinsics, x.size(3), x.size(2));

        // Encoder
        auto skip1 = enc1(x, normalized_intrinsics);
        auto skip2 = enc2(skip1, normalized_intrinsics);
        auto skip3 = enc3(skip2, normalized_intrinsics);
        auto skip4 = enc4(skip3, normalized_intrinsics);

        // Bottleneck
        auto x_bottleneck = bottleneck(skip4, normalized_intrinsics);

        // Decoder with attention
        x = dec4(x_bottleneck, skip4, normalized_intrinsics);
        x = att4(x);

        x = dec3(x, skip3, normalized_intrinsics);
        x = att3(x);

        x = dec2(x, skip2, normalized_intrinsics);
        x = att2(x);

        x = dec1(x, skip1, normalized_intrinsics);
        x = att1(x);

        // Output
        x = out_conv(x);
        x = torch::sigmoid(x) * max_depth;

        return x;
    }

private:
    torch::Tensor normalizeCameraIntrinsics(torch::Tensor intrinsics, int width, int height) {
        auto normalized = intrinsics.clone();
        normalized.index_put_({torch::indexing::Slice(), 0},
            intrinsics.index({torch::indexing::Slice(), 0}) / width);
        normalized.index_put_({torch::indexing::Slice(), 1},
            intrinsics.index({torch::indexing::Slice(), 1}) / height);
        normalized.index_put_({torch::indexing::Slice(), 2},
            (intrinsics.index({torch::indexing::Slice(), 2}) / width) * 2.0f - 1.0f);
        normalized.index_put_({torch::indexing::Slice(), 3},
            (intrinsics.index({torch::indexing::Slice(), 3}) / height) * 2.0f - 1.0f);
        return normalized;
    }
};
TORCH_MODULE(IntrinsicsAttentionUNet);

} // namespace camera_aware_depth

#endif // INTRINSICS_UNET_H
