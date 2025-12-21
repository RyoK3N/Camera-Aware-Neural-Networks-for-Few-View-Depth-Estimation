#ifndef GEOMETRY_AWARE_NETWORK_H
#define GEOMETRY_AWARE_NETWORK_H

#include <torch/torch.h>
#include "../layers/film_layer.h"
#include "../layers/spatial_attention.h"
#include "../layers/pcl_layer.h"

namespace camera_aware_depth {

/**
 * @brief Ray-Enhanced Convolution Block
 *
 * Convolution block that incorporates ray direction information.
 * Concatenates ray directions with features for geometric reasoning.
 */
struct RayEnhancedConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};
    FiLMLayer film{nullptr};

    int out_channels_;

    RayEnhancedConvImpl(int in_channels, int out_channels, int camera_dim = 4, bool use_rays = true)
        : out_channels_(out_channels) {

        // If using rays, add 3 channels for ray directions (x, y, z)
        int total_in_channels = use_rays ? in_channels + 3 : in_channels;

        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(total_in_channels, out_channels, 3).padding(1).bias(false)
        ));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(false)
        ));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

        // FiLM for camera conditioning
        film = register_module("film", FiLMLayer(camera_dim, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_params,
                         torch::optional<torch::Tensor> ray_directions = torch::nullopt) {
        // Concatenate ray directions if provided
        if (ray_directions.has_value()) {
            x = torch::cat({x, ray_directions.value()}, 1);
        }

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
TORCH_MODULE(RayEnhancedConv);

/**
 * @brief Geometry-Aware Encoder Block
 *
 * Encoder block with ray enhancement, FiLM conditioning, and optional attention
 */
struct GeometryEncoderBlockImpl : torch::nn::Module {
    torch::nn::MaxPool2d pool{nullptr};
    RayEnhancedConv conv{nullptr};
    CBAM attention{nullptr};

    bool use_attention_;

    GeometryEncoderBlockImpl(int in_channels, int out_channels, int camera_dim = 4,
                            bool use_rays = true, bool use_attention = true)
        : use_attention_(use_attention) {

        pool = register_module("pool", torch::nn::MaxPool2d(2));
        conv = register_module("conv", RayEnhancedConv(in_channels, out_channels, camera_dim, use_rays));

        if (use_attention) {
            attention = register_module("attention", CBAM(out_channels));
        }
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_params,
                         torch::optional<torch::Tensor> ray_directions = torch::nullopt) {
        x = pool(x);
        x = conv(x, camera_params, ray_directions);

        if (use_attention_ && !attention.is_empty()) {
            x = attention(x);
        }

        return x;
    }
};
TORCH_MODULE(GeometryEncoderBlock);

/**
 * @brief Geometry-Aware Decoder Block
 *
 * Decoder block with PCL, FiLM conditioning, and attention
 */
struct GeometryDecoderBlockImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d up{nullptr};
    RayEnhancedConv conv{nullptr};
    PerspectiveCorrectionLayer pcl{nullptr};
    CBAM attention{nullptr};

    bool use_pcl_;
    bool use_attention_;

    GeometryDecoderBlockImpl(int in_channels, int out_channels, int camera_dim = 4,
                            bool use_pcl = true, bool use_attention = true)
        : use_pcl_(use_pcl), use_attention_(use_attention) {

        up = register_module("up", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 2).stride(2)
        ));
        conv = register_module("conv", RayEnhancedConv(in_channels, out_channels, camera_dim, false));

        if (use_pcl) {
            pcl = register_module("pcl", PerspectiveCorrectionLayer(out_channels, camera_dim));
        }

        if (use_attention) {
            attention = register_module("attention", CBAM(out_channels));
        }
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor skip, torch::Tensor camera_params,
                         torch::optional<torch::Tensor> ray_directions = torch::nullopt) {
        x = up(x);

        // Apply perspective correction to upsampled features
        if (use_pcl_ && !pcl.is_empty()) {
            x = pcl(x, camera_params, ray_directions);
        }

        // Handle size mismatch
        int diff_h = skip.size(2) - x.size(2);
        int diff_w = skip.size(3) - x.size(3);

        if (diff_h > 0 || diff_w > 0) {
            x = torch::nn::functional::pad(x,
                torch::nn::functional::PadFuncOptions({diff_w / 2, diff_w - diff_w / 2,
                                                       diff_h / 2, diff_h - diff_h / 2}));
        }

        // Concatenate with skip connection
        x = torch::cat({skip, x}, 1);
        x = conv(x, camera_params);

        // Apply attention
        if (use_attention_ && !attention.is_empty()) {
            x = attention(x);
        }

        return x;
    }
};
TORCH_MODULE(GeometryDecoderBlock);

/**
 * @brief Complete Geometry-Aware Network for Depth Estimation
 *
 * Combines all geometric priors and conditioning mechanisms:
 * 1. Ray direction integration in encoder
 * 2. FiLM conditioning on camera intrinsics throughout the network
 * 3. Perspective Correction Layers (PCL) in decoder
 * 4. CBAM attention modules for feature refinement
 *
 * Based on:
 * - Camera-Aware Neural Networks (research objective)
 * - FiLM: Perez et al., AAAI 2018
 * - Spatial Transformer Networks: Jaderberg et al., NIPS 2015
 * - CBAM: Woo et al., ECCV 2018
 *
 * Architecture Overview:
 * Input: RGB (B, 3, H, W) + Ray Directions (B, 3, H, W) + Camera Intrinsics (B, 4)
 *
 * Encoder:
 *   - Level 1: RGB + Rays → Features (64 ch)
 *   - Level 2-5: Downsampling with ray enhancement and FiLM
 *   - Each level: Conv → FiLM → Attention
 *
 * Decoder:
 *   - Level 5-1: Upsampling with PCL and FiLM
 *   - Each level: Upsample → PCL → Concat Skip → Conv → FiLM → Attention
 *
 * Output: Depth map (B, 1, H, W)
 */
struct GeometryAwareNetworkImpl : torch::nn::Module {
    // Initial ray-enhanced convolution
    RayEnhancedConv enc1{nullptr};

    // Encoder pathway
    GeometryEncoderBlock enc2{nullptr};
    GeometryEncoderBlock enc3{nullptr};
    GeometryEncoderBlock enc4{nullptr};
    GeometryEncoderBlock enc5{nullptr};

    // Bottleneck
    GeometryEncoderBlock bottleneck{nullptr};

    // Decoder pathway
    GeometryDecoderBlock dec5{nullptr};
    GeometryDecoderBlock dec4{nullptr};
    GeometryDecoderBlock dec3{nullptr};
    GeometryDecoderBlock dec2{nullptr};
    GeometryDecoderBlock dec1{nullptr};

    // Output head
    torch::nn::Conv2d out_conv{nullptr};

    // Configuration
    float max_depth_;
    int camera_dim_;
    int init_features_;

    /**
     * @brief Construct Geometry-Aware Network
     *
     * @param in_channels Input image channels (default: 3 for RGB)
     * @param init_features Initial feature dimension (default: 64)
     * @param camera_dim Camera parameter dimension (default: 4 for [fx, fy, cx, cy])
     * @param max_depth Maximum depth value for scaling (default: 10.0)
     * @param use_pcl Whether to use Perspective Correction Layers (default: true)
     * @param use_attention Whether to use attention modules (default: true)
     */
    GeometryAwareNetworkImpl(int in_channels = 3,
                            int init_features = 64,
                            int camera_dim = 4,
                            float max_depth = 10.0f,
                            bool use_pcl = true,
                            bool use_attention = true)
        : max_depth_(max_depth), camera_dim_(camera_dim), init_features_(init_features) {

        // Initial encoder with ray directions
        enc1 = register_module("enc1",
            RayEnhancedConv(in_channels, init_features, camera_dim, true));

        // Encoder pathway (with downsampling)
        // Note: enc2-enc5 don't use rays directly (already encoded in features)
        enc2 = register_module("enc2",
            GeometryEncoderBlock(init_features, init_features * 2, camera_dim, false, use_attention));
        enc3 = register_module("enc3",
            GeometryEncoderBlock(init_features * 2, init_features * 4, camera_dim, false, use_attention));
        enc4 = register_module("enc4",
            GeometryEncoderBlock(init_features * 4, init_features * 8, camera_dim, false, use_attention));
        enc5 = register_module("enc5",
            GeometryEncoderBlock(init_features * 8, init_features * 16, camera_dim, false, use_attention));

        // Bottleneck
        bottleneck = register_module("bottleneck",
            GeometryEncoderBlock(init_features * 16, init_features * 32, camera_dim, false, use_attention));

        // Decoder pathway (with upsampling and PCL)
        dec5 = register_module("dec5",
            GeometryDecoderBlock(init_features * 32, init_features * 16, camera_dim, use_pcl, use_attention));
        dec4 = register_module("dec4",
            GeometryDecoderBlock(init_features * 16, init_features * 8, camera_dim, use_pcl, use_attention));
        dec3 = register_module("dec3",
            GeometryDecoderBlock(init_features * 8, init_features * 4, camera_dim, use_pcl, use_attention));
        dec2 = register_module("dec2",
            GeometryDecoderBlock(init_features * 4, init_features * 2, camera_dim, use_pcl, use_attention));
        dec1 = register_module("dec1",
            GeometryDecoderBlock(init_features * 2, init_features, camera_dim, use_pcl, use_attention));

        // Output convolution
        out_conv = register_module("out_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(init_features, 1, 1)
        ));
    }

    /**
     * @brief Forward pass
     *
     * @param rgb Input RGB image (B, 3, H, W)
     * @param ray_directions Ray directions (B, 3, H, W)
     * @param camera_intrinsics Camera intrinsics (B, 4) - [fx, fy, cx, cy]
     * @return Predicted depth map (B, 1, H, W)
     */
    torch::Tensor forward(torch::Tensor rgb,
                         torch::Tensor ray_directions,
                         torch::Tensor camera_intrinsics) {

        // Normalize camera intrinsics
        auto norm_intrinsics = normalizeCameraIntrinsics(camera_intrinsics,
                                                         rgb.size(3), rgb.size(2));

        // Encoder: progressively downsample and encode geometry
        auto skip1 = enc1(rgb, norm_intrinsics, ray_directions);
        auto skip2 = enc2(skip1, norm_intrinsics);
        auto skip3 = enc3(skip2, norm_intrinsics);
        auto skip4 = enc4(skip3, norm_intrinsics);
        auto skip5 = enc5(skip4, norm_intrinsics);

        // Bottleneck: deepest feature representation
        auto x = bottleneck(skip5, norm_intrinsics);

        // Decoder: progressively upsample with perspective correction
        // Pass downsampled ray directions for PCL
        x = dec5(x, skip5, norm_intrinsics, getDownsampledRays(ray_directions, 16));
        x = dec4(x, skip4, norm_intrinsics, getDownsampledRays(ray_directions, 8));
        x = dec3(x, skip3, norm_intrinsics, getDownsampledRays(ray_directions, 4));
        x = dec2(x, skip2, norm_intrinsics, getDownsampledRays(ray_directions, 2));
        x = dec1(x, skip1, norm_intrinsics, ray_directions);

        // Output: predict depth with sigmoid activation
        x = out_conv(x);
        x = torch::sigmoid(x) * max_depth_;

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

    /**
     * @brief Get memory usage estimate in MB
     */
    float estimate_memory_mb(int batch_size, int height, int width) {
        // Rough estimate based on feature maps
        int64_t pixels = batch_size * height * width;

        // Encoder feature maps
        int64_t enc_memory = pixels * (init_features_ +           // enc1
                                       init_features_ * 2 / 4 +   // enc2
                                       init_features_ * 4 / 16 +  // enc3
                                       init_features_ * 8 / 64 +  // enc4
                                       init_features_ * 16 / 256);// enc5

        // Decoder feature maps (similar)
        int64_t dec_memory = enc_memory;

        // Parameters
        int64_t param_memory = count_parameters();

        // Total in bytes (4 bytes per float32)
        int64_t total_bytes = (enc_memory + dec_memory + param_memory) * 4;

        return total_bytes / (1024.0f * 1024.0f);
    }

private:
    /**
     * @brief Normalize camera intrinsics for stable conditioning
     */
    torch::Tensor normalizeCameraIntrinsics(torch::Tensor intrinsics, int width, int height) {
        auto normalized = intrinsics.clone();

        // Normalize focal lengths by image dimensions
        normalized.index_put_({torch::indexing::Slice(), 0},
            intrinsics.index({torch::indexing::Slice(), 0}) / width);
        normalized.index_put_({torch::indexing::Slice(), 1},
            intrinsics.index({torch::indexing::Slice(), 1}) / height);

        // Normalize principal point to [-1, 1]
        normalized.index_put_({torch::indexing::Slice(), 2},
            (intrinsics.index({torch::indexing::Slice(), 2}) / width) * 2.0f - 1.0f);
        normalized.index_put_({torch::indexing::Slice(), 3},
            (intrinsics.index({torch::indexing::Slice(), 3}) / height) * 2.0f - 1.0f);

        return normalized;
    }

    /**
     * @brief Downsample ray directions for multi-scale processing
     */
    torch::Tensor getDownsampledRays(torch::Tensor rays, int factor) {
        if (factor <= 1) return rays;

        return torch::nn::functional::avg_pool2d(rays,
            torch::nn::functional::AvgPool2dFuncOptions(factor).stride(factor));
    }
};
TORCH_MODULE(GeometryAwareNetwork);

/**
 * @brief Lightweight Geometry-Aware Network
 *
 * Smaller version for faster training and inference.
 * Uses fewer features and shallower architecture.
 */
struct LightweightGeometryNetworkImpl : torch::nn::Module {
    RayEnhancedConv enc1{nullptr};
    GeometryEncoderBlock enc2{nullptr};
    GeometryEncoderBlock enc3{nullptr};
    GeometryEncoderBlock enc4{nullptr};

    GeometryEncoderBlock bottleneck{nullptr};

    GeometryDecoderBlock dec4{nullptr};
    GeometryDecoderBlock dec3{nullptr};
    GeometryDecoderBlock dec2{nullptr};
    GeometryDecoderBlock dec1{nullptr};

    torch::nn::Conv2d out_conv{nullptr};

    float max_depth_;

    LightweightGeometryNetworkImpl(int in_channels = 3,
                                   int init_features = 32,
                                   int camera_dim = 4,
                                   float max_depth = 10.0f)
        : max_depth_(max_depth) {

        enc1 = register_module("enc1", RayEnhancedConv(in_channels, init_features, camera_dim, true));
        enc2 = register_module("enc2", GeometryEncoderBlock(init_features, init_features * 2, camera_dim, false, true));
        enc3 = register_module("enc3", GeometryEncoderBlock(init_features * 2, init_features * 4, camera_dim, false, true));
        enc4 = register_module("enc4", GeometryEncoderBlock(init_features * 4, init_features * 8, camera_dim, false, true));

        bottleneck = register_module("bottleneck", GeometryEncoderBlock(init_features * 8, init_features * 16, camera_dim, false, true));

        dec4 = register_module("dec4", GeometryDecoderBlock(init_features * 16, init_features * 8, camera_dim, true, true));
        dec3 = register_module("dec3", GeometryDecoderBlock(init_features * 8, init_features * 4, camera_dim, true, true));
        dec2 = register_module("dec2", GeometryDecoderBlock(init_features * 4, init_features * 2, camera_dim, true, true));
        dec1 = register_module("dec1", GeometryDecoderBlock(init_features * 2, init_features, camera_dim, true, true));

        out_conv = register_module("out_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(init_features, 1, 1)
        ));
    }

    torch::Tensor forward(torch::Tensor rgb, torch::Tensor ray_directions, torch::Tensor camera_intrinsics) {
        auto norm_intrinsics = normalizeCameraIntrinsics(camera_intrinsics, rgb.size(3), rgb.size(2));

        auto skip1 = enc1(rgb, norm_intrinsics, ray_directions);
        auto skip2 = enc2(skip1, norm_intrinsics);
        auto skip3 = enc3(skip2, norm_intrinsics);
        auto skip4 = enc4(skip3, norm_intrinsics);

        auto x = bottleneck(skip4, norm_intrinsics);

        x = dec4(x, skip4, norm_intrinsics, torch::nn::functional::avg_pool2d(ray_directions, torch::nn::functional::AvgPool2dFuncOptions(8).stride(8)));
        x = dec3(x, skip3, norm_intrinsics, torch::nn::functional::avg_pool2d(ray_directions, torch::nn::functional::AvgPool2dFuncOptions(4).stride(4)));
        x = dec2(x, skip2, norm_intrinsics, torch::nn::functional::avg_pool2d(ray_directions, torch::nn::functional::AvgPool2dFuncOptions(2).stride(2)));
        x = dec1(x, skip1, norm_intrinsics, ray_directions);

        x = out_conv(x);
        x = torch::sigmoid(x) * max_depth_;

        return x;
    }

private:
    torch::Tensor normalizeCameraIntrinsics(torch::Tensor intrinsics, int width, int height) {
        auto normalized = intrinsics.clone();
        normalized.index_put_({torch::indexing::Slice(), 0}, intrinsics.index({torch::indexing::Slice(), 0}) / width);
        normalized.index_put_({torch::indexing::Slice(), 1}, intrinsics.index({torch::indexing::Slice(), 1}) / height);
        normalized.index_put_({torch::indexing::Slice(), 2}, (intrinsics.index({torch::indexing::Slice(), 2}) / width) * 2.0f - 1.0f);
        normalized.index_put_({torch::indexing::Slice(), 3}, (intrinsics.index({torch::indexing::Slice(), 3}) / height) * 2.0f - 1.0f);
        return normalized;
    }
};
TORCH_MODULE(LightweightGeometryNetwork);

} // namespace camera_aware_depth

#endif // GEOMETRY_AWARE_NETWORK_H
