#ifndef BASELINE_UNET_H
#define BASELINE_UNET_H

#include <torch/torch.h>

namespace camera_aware_depth {

/**
 * @brief Double Convolution Block
 *
 * Standard U-Net building block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
 * Based on original U-Net paper (Ronneberger et al., 2015)
 */
struct DoubleConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};

    DoubleConvImpl(int in_channels, int out_channels) {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).bias(false)
        ));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(false)
        ));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1(x);
        x = bn1(x);
        x = torch::relu(x);

        x = conv2(x);
        x = bn2(x);
        x = torch::relu(x);

        return x;
    }
};
TORCH_MODULE(DoubleConv);

/**
 * @brief Encoder Block
 *
 * MaxPool -> DoubleConv
 */
struct EncoderBlockImpl : torch::nn::Module {
    torch::nn::MaxPool2d pool{nullptr};
    DoubleConv conv{nullptr};

    EncoderBlockImpl(int in_channels, int out_channels) {
        pool = register_module("pool", torch::nn::MaxPool2d(2));
        conv = register_module("conv", DoubleConv(in_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = pool(x);
        x = conv(x);
        return x;
    }
};
TORCH_MODULE(EncoderBlock);

/**
 * @brief Decoder Block
 *
 * UpConv -> Concatenate with skip -> DoubleConv
 */
struct DecoderBlockImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d up{nullptr};
    DoubleConv conv{nullptr};

    DecoderBlockImpl(int in_channels, int out_channels) {
        up = register_module("up", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 2).stride(2)
        ));
        conv = register_module("conv", DoubleConv(in_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor skip) {
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
        x = conv(x);

        return x;
    }
};
TORCH_MODULE(DecoderBlock);

/**
 * @brief Baseline U-Net for Depth Estimation
 *
 * Architecture based on:
 * - Original U-Net (Ronneberger et al., MICCAI 2015)
 * - Adapted for monocular depth estimation
 *
 * Input: RGB image (3 channels)
 * Output: Depth map (1 channel)
 *
 * Features:
 * - 4-level encoder-decoder architecture
 * - Skip connections for preserving spatial information
 * - Batch normalization for training stability
 * - Sigmoid activation for normalized depth output
 */
struct BaselineUNetImpl : torch::nn::Module {
    // Encoder
    DoubleConv enc1{nullptr};
    EncoderBlock enc2{nullptr};
    EncoderBlock enc3{nullptr};
    EncoderBlock enc4{nullptr};

    // Bottleneck
    EncoderBlock bottleneck{nullptr};

    // Decoder
    DecoderBlock dec4{nullptr};
    DecoderBlock dec3{nullptr};
    DecoderBlock dec2{nullptr};
    DecoderBlock dec1{nullptr};

    // Output
    torch::nn::Conv2d out_conv{nullptr};

    // Configuration
    float max_depth;

    BaselineUNetImpl(int in_channels = 3, int init_features = 64, float max_depth_value = 10.0f)
        : max_depth(max_depth_value) {

        // Encoder pathway
        enc1 = register_module("enc1", DoubleConv(in_channels, init_features));
        enc2 = register_module("enc2", EncoderBlock(init_features, init_features * 2));
        enc3 = register_module("enc3", EncoderBlock(init_features * 2, init_features * 4));
        enc4 = register_module("enc4", EncoderBlock(init_features * 4, init_features * 8));

        // Bottleneck
        bottleneck = register_module("bottleneck", EncoderBlock(init_features * 8, init_features * 16));

        // Decoder pathway
        dec4 = register_module("dec4", DecoderBlock(init_features * 16, init_features * 8));
        dec3 = register_module("dec3", DecoderBlock(init_features * 8, init_features * 4));
        dec2 = register_module("dec2", DecoderBlock(init_features * 4, init_features * 2));
        dec1 = register_module("dec1", DecoderBlock(init_features * 2, init_features));

        // Output layer: 1x1 convolution to get depth map
        out_conv = register_module("out_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(init_features, 1, 1)
        ));
    }

    /**
     * @brief Forward pass
     *
     * @param x Input RGB image (B, 3, H, W)
     * @return Depth map (B, 1, H, W) in range [0, max_depth]
     */
    torch::Tensor forward(torch::Tensor x) {
        // Encoder with skip connections
        auto skip1 = enc1(x);
        auto skip2 = enc2(skip1);
        auto skip3 = enc3(skip2);
        auto skip4 = enc4(skip3);

        // Bottleneck
        auto x_bottleneck = bottleneck(skip4);

        // Decoder with skip connections
        x = dec4(x_bottleneck, skip4);
        x = dec3(x, skip3);
        x = dec2(x, skip2);
        x = dec1(x, skip1);

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
};
TORCH_MODULE(BaselineUNet);

} // namespace camera_aware_depth

#endif // BASELINE_UNET_H
