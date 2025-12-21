#ifndef SPATIAL_ATTENTION_H
#define SPATIAL_ATTENTION_H

#include <torch/torch.h>

namespace camera_aware_depth {

/**
 * @brief Channel Attention Module
 *
 * Based on CBAM (Convolutional Block Attention Module)
 * Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
 * Paper: https://arxiv.org/abs/1807.06521
 *
 * Channel attention adaptively recalibrates channel-wise feature responses by
 * modeling interdependencies between channels using both max-pooling and
 * average-pooling spatial information.
 *
 * Formula:
 * M_c = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
 * F' = M_c ⊙ F
 */
struct ChannelAttentionImpl : torch::nn::Module {
    torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
    torch::nn::AdaptiveMaxPool2d max_pool{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    int channels_;
    int reduction_ratio_;

    /**
     * @brief Construct Channel Attention Module
     *
     * @param channels Number of input channels
     * @param reduction_ratio Channel reduction ratio for MLP (default: 16)
     */
    ChannelAttentionImpl(int channels, int reduction_ratio = 16)
        : channels_(channels), reduction_ratio_(reduction_ratio) {

        avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(1));
        max_pool = register_module("max_pool", torch::nn::AdaptiveMaxPool2d(1));

        int reduced_channels = channels / reduction_ratio;
        if (reduced_channels < 1) reduced_channels = 1;

        // Shared MLP: two fully connected layers
        fc1 = register_module("fc1", torch::nn::Linear(channels, reduced_channels));
        fc2 = register_module("fc2", torch::nn::Linear(reduced_channels, channels));
    }

    /**
     * @brief Forward pass
     *
     * @param x Input features (B, C, H, W)
     * @return Channel attention map (B, C, 1, 1)
     */
    torch::Tensor forward(torch::Tensor x) {
        int batch_size = x.size(0);

        // Average pooling branch
        auto avg_out = avg_pool(x).view({batch_size, channels_});
        avg_out = torch::relu(fc1(avg_out));
        avg_out = fc2(avg_out);

        // Max pooling branch
        auto max_out = max_pool(x).view({batch_size, channels_});
        max_out = torch::relu(fc1(max_out));
        max_out = fc2(max_out);

        // Combine and apply sigmoid
        auto attention = torch::sigmoid(avg_out + max_out);
        return attention.view({batch_size, channels_, 1, 1});
    }
};
TORCH_MODULE(ChannelAttention);

/**
 * @brief Spatial Attention Module
 *
 * Based on CBAM (Convolutional Block Attention Module)
 * Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
 *
 * Spatial attention focuses on "where" is an informative part by utilizing
 * inter-spatial relationships of features. It generates a spatial attention map
 * by leveraging both average-pooled and max-pooled features along the channel axis.
 *
 * Formula:
 * M_s = σ(Conv([AvgPool(F); MaxPool(F)]))
 * F' = M_s ⊙ F
 */
struct SpatialAttentionImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};

    SpatialAttentionImpl(int kernel_size = 7) {
        // Convolution to process concatenated pooled features
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 1, kernel_size)
                .padding(kernel_size / 2)
                .bias(false)
        ));
    }

    /**
     * @brief Forward pass
     *
     * @param x Input features (B, C, H, W)
     * @return Spatial attention map (B, 1, H, W)
     */
    torch::Tensor forward(torch::Tensor x) {
        // Average pooling along channel dimension
        auto avg_out = torch::mean(x, /*dim=*/1, /*keepdim=*/true);  // (B, 1, H, W)

        // Max pooling along channel dimension
        auto max_out = std::get<0>(torch::max(x, /*dim=*/1, /*keepdim=*/true));  // (B, 1, H, W)

        // Concatenate along channel dimension
        auto concat = torch::cat({avg_out, max_out}, /*dim=*/1);  // (B, 2, H, W)

        // Apply convolution and sigmoid
        auto attention = torch::sigmoid(conv(concat));  // (B, 1, H, W)

        return attention;
    }
};
TORCH_MODULE(SpatialAttention);

/**
 * @brief CBAM (Convolutional Block Attention Module)
 *
 * Combines channel and spatial attention in sequence.
 * Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
 * Paper: https://arxiv.org/abs/1807.06521
 *
 * CBAM sequentially applies channel and spatial attention:
 * 1. Channel attention: F' = M_c(F) ⊙ F
 * 2. Spatial attention: F'' = M_s(F') ⊙ F'
 *
 * This module has been shown to improve performance on various vision tasks
 * including depth estimation, semantic segmentation, and object detection.
 */
struct CBAMImpl : torch::nn::Module {
    ChannelAttention channel_attention{nullptr};
    SpatialAttention spatial_attention{nullptr};

    /**
     * @brief Construct CBAM
     *
     * @param channels Number of input channels
     * @param reduction_ratio Channel reduction ratio (default: 16)
     * @param spatial_kernel Spatial attention kernel size (default: 7)
     */
    CBAMImpl(int channels, int reduction_ratio = 16, int spatial_kernel = 7) {
        channel_attention = register_module("channel_attention",
            ChannelAttention(channels, reduction_ratio));
        spatial_attention = register_module("spatial_attention",
            SpatialAttention(spatial_kernel));
    }

    /**
     * @brief Forward pass
     *
     * @param x Input features (B, C, H, W)
     * @return Attention-refined features (B, C, H, W)
     */
    torch::Tensor forward(torch::Tensor x) {
        // Apply channel attention
        auto channel_att = channel_attention(x);
        x = x * channel_att;

        // Apply spatial attention
        auto spatial_att = spatial_attention(x);
        x = x * spatial_att;

        return x;
    }

    /**
     * @brief Get attention maps for visualization
     *
     * Returns both channel and spatial attention maps
     */
    std::pair<torch::Tensor, torch::Tensor> getAttentionMaps(torch::Tensor x) {
        auto channel_att = channel_attention(x);
        auto x_after_channel = x * channel_att;
        auto spatial_att = spatial_attention(x_after_channel);

        return std::make_pair(channel_att, spatial_att);
    }
};
TORCH_MODULE(CBAM);

/**
 * @brief Depth-Specific Spatial Attention
 *
 * Enhanced spatial attention specifically designed for depth estimation.
 * Incorporates edge information and multi-scale context.
 *
 * Based on:
 * - "Self-attention Presents Low-dimensional Knowledge Graph Embeddings for Link Prediction"
 * - "Attention-based Context Aggregation Network for Monocular Depth Estimation" (Chen et al.)
 *
 * This module is particularly useful for depth estimation as it:
 * - Emphasizes depth discontinuities (edges)
 * - Aggregates multi-scale spatial context
 * - Maintains fine-grained spatial details
 */
struct DepthSpatialAttentionImpl : torch::nn::Module {
    torch::nn::Conv2d edge_conv{nullptr};
    torch::nn::Conv2d context_conv{nullptr};
    torch::nn::Conv2d fusion_conv{nullptr};
    SpatialAttention base_attention{nullptr};

    DepthSpatialAttentionImpl(int channels, int kernel_size = 7) {
        // Edge detection branch
        edge_conv = register_module("edge_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, 1, 3).padding(1).bias(false)
        ));

        // Multi-scale context branch
        context_conv = register_module("context_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, 1, kernel_size)
                .padding(kernel_size / 2)
                .dilation(2)  // Dilated conv for larger receptive field
                .bias(false)
        ));

        // Fusion convolution
        fusion_conv = register_module("fusion_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 1, 1).bias(false)
        ));

        // Base spatial attention
        base_attention = register_module("base_attention",
            SpatialAttention(kernel_size));
    }

    /**
     * @brief Forward pass
     *
     * @param x Input features (B, C, H, W)
     * @return Depth-aware spatial attention map (B, 1, H, W)
     */
    torch::Tensor forward(torch::Tensor x) {
        // Base attention (avg + max pooling)
        auto base_att = base_attention(x);

        // Edge-aware attention
        auto edge_att = torch::sigmoid(edge_conv(x));

        // Context-aware attention with larger receptive field
        auto context_att = torch::sigmoid(context_conv(x));

        // Fuse all attention sources
        auto fused = torch::cat({base_att, edge_att, context_att}, /*dim=*/1);
        auto attention = torch::sigmoid(fusion_conv(fused));

        return attention;
    }
};
TORCH_MODULE(DepthSpatialAttention);

/**
 * @brief Camera-Aware Spatial Attention
 *
 * Spatial attention conditioned on camera parameters (intrinsics).
 * This allows the attention mechanism to adapt based on camera properties
 * like focal length and field of view.
 *
 * Useful for handling multi-camera scenarios or varying camera parameters.
 */
struct CameraAwareSpatialAttentionImpl : torch::nn::Module {
    torch::nn::Linear camera_fc{nullptr};
    torch::nn::Conv2d spatial_conv{nullptr};
    torch::nn::Conv2d fusion_conv{nullptr};

    int channels_;

    CameraAwareSpatialAttentionImpl(int channels, int camera_dim = 4, int kernel_size = 7)
        : channels_(channels) {

        // Camera parameter embedding
        camera_fc = register_module("camera_fc", torch::nn::Linear(camera_dim, channels));

        // Spatial convolution
        spatial_conv = register_module("spatial_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, channels, kernel_size)
                .padding(kernel_size / 2)
                .bias(false)
        ));

        // Fusion to generate final attention
        fusion_conv = register_module("fusion_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels * 2, 1, 1)
        ));
    }

    /**
     * @brief Forward pass
     *
     * @param x Input features (B, C, H, W)
     * @param camera_params Camera parameters (B, camera_dim)
     * @return Camera-aware spatial attention map (B, 1, H, W)
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor camera_params) {
        int batch_size = x.size(0);
        int height = x.size(2);
        int width = x.size(3);

        // Spatial pooling
        auto avg_out = torch::mean(x, /*dim=*/1, /*keepdim=*/true);
        auto max_out = std::get<0>(torch::max(x, /*dim=*/1, /*keepdim=*/true));
        auto pooled = torch::cat({avg_out, max_out}, /*dim=*/1);

        // Spatial features
        auto spatial_feat = spatial_conv(pooled);  // (B, C, H, W)

        // Camera conditioning
        auto camera_feat = torch::relu(camera_fc(camera_params));  // (B, C)
        camera_feat = camera_feat.view({batch_size, channels_, 1, 1});
        camera_feat = camera_feat.expand({batch_size, channels_, height, width});

        // Fuse spatial and camera features
        auto fused = torch::cat({spatial_feat, camera_feat}, /*dim=*/1);
        auto attention = torch::sigmoid(fusion_conv(fused));

        return attention;
    }
};
TORCH_MODULE(CameraAwareSpatialAttention);

} // namespace camera_aware_depth

#endif // SPATIAL_ATTENTION_H
