#include <iostream>
#include <iomanip>
#include <chrono>
#include <torch/torch.h>
#include "../src/models/baseline_unet.h"
#include "../src/models/intrinsics_unet.h"
#include "../src/models/geometry_aware_network.h"
#include "../src/loss/depth_loss.h"
#include "../src/layers/film_layer.h"
#include "../src/layers/spatial_attention.h"
#include "../src/layers/pcl_layer.h"

using namespace camera_aware_depth;

/**
 * @brief Test result structure
 */
struct TestResult {
    std::string test_name;
    bool passed;
    std::string message;
    double duration_ms;
};

/**
 * @brief Print test result with color coding
 */
void printTestResult(const TestResult& result) {
    std::string status = result.passed ? "[PASS]" : "[FAIL]";
    std::string color = result.passed ? "\033[32m" : "\033[31m";
    std::string reset = "\033[0m";

    std::cout << color << std::setw(8) << status << reset << " "
              << std::setw(50) << std::left << result.test_name << " "
              << std::fixed << std::setprecision(2) << result.duration_ms << " ms";

    if (!result.message.empty()) {
        std::cout << " - " << result.message;
    }
    std::cout << std::endl;
}

/**
 * @brief Timer helper class
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Test FiLM Layer
 */
TestResult testFiLMLayer() {
    Timer timer;
    TestResult result{"FiLM Layer", false, "", 0.0};

    try {
        // Create FiLM layer
        int camera_dim = 4;
        int feature_channels = 64;
        auto film = FiLMLayer(camera_dim, feature_channels);

        // Create test inputs
        auto features = torch::randn({2, 64, 32, 32});  // (B, C, H, W)
        auto camera_params = torch::randn({2, 4});      // (B, camera_dim)

        // Forward pass
        auto output = film(features, camera_params);

        // Check output shape
        if (output.sizes() != features.sizes()) {
            result.message = "Output shape mismatch";
            return result;
        }

        // Check that output is different from input (modulation applied)
        auto diff = torch::abs(output - features).mean().item<float>();
        if (diff < 1e-6) {
            result.message = "No modulation applied";
            return result;
        }

        // Test get_modulation_params
        auto [gamma, beta] = film->get_modulation_params(camera_params);
        if (gamma.size(0) != 2 || gamma.size(1) != 64) {
            result.message = "Gamma shape incorrect";
            return result;
        }

        result.passed = true;
        result.message = "All checks passed";

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test CBAM Attention
 */
TestResult testCBAM() {
    Timer timer;
    TestResult result{"CBAM Attention", false, "", 0.0};

    try {
        int channels = 64;
        auto cbam = CBAM(channels);

        auto input = torch::randn({2, 64, 32, 32});
        auto output = cbam(input);

        // Check shape
        if (output.sizes() != input.sizes()) {
            result.message = "Output shape mismatch";
            return result;
        }

        // Check that attention is applied
        auto diff = torch::abs(output - input).mean().item<float>();
        if (diff < 1e-6) {
            result.message = "No attention applied";
            return result;
        }

        // Test getAttentionMaps
        auto [channel_att, spatial_att] = cbam->getAttentionMaps(input);
        if (channel_att.size(1) != channels || spatial_att.size(1) != 1) {
            result.message = "Attention map shapes incorrect";
            return result;
        }

        result.passed = true;
        result.message = "All checks passed";

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Perspective Correction Layer
 */
TestResult testPCL() {
    Timer timer;
    TestResult result{"Perspective Correction Layer", false, "", 0.0};

    try {
        int feature_channels = 64;
        int camera_dim = 4;
        auto pcl = PerspectiveCorrectionLayer(feature_channels, camera_dim);

        auto features = torch::randn({2, 64, 32, 32});
        auto camera_intrinsics = torch::randn({2, 4});
        auto ray_directions = torch::randn({2, 3, 32, 32});

        // Test 2D transformation
        auto output = pcl(features, camera_intrinsics);

        if (output.sizes() != features.sizes()) {
            result.message = "Output shape mismatch (2D)";
            return result;
        }

        // Test 3D transformation
        auto output_3d = pcl->forward3D(features, camera_intrinsics, ray_directions);

        if (output_3d.sizes() != features.sizes()) {
            result.message = "Output shape mismatch (3D)";
            return result;
        }

        result.passed = true;
        result.message = "All checks passed";

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Baseline U-Net
 */
TestResult testBaselineUNet() {
    Timer timer;
    TestResult result{"Baseline U-Net", false, "", 0.0};

    try {
        auto model = BaselineUNet(3, 64, 10.0f);

        auto input = torch::randn({2, 3, 256, 256});
        auto output = model(input);

        // Check output shape
        if (output.size(0) != 2 || output.size(1) != 1 ||
            output.size(2) != 256 || output.size(3) != 256) {
            result.message = "Output shape incorrect";
            return result;
        }

        // Check output range [0, 10]
        auto min_val = output.min().item<float>();
        auto max_val = output.max().item<float>();
        if (min_val < 0.0f || max_val > 10.0f) {
            result.message = "Output out of range";
            return result;
        }

        // Check parameter count
        auto param_count = model->count_parameters();
        if (param_count == 0) {
            result.message = "No parameters";
            return result;
        }

        result.passed = true;
        result.message = std::to_string(param_count) + " parameters";

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Intrinsics-Conditioned U-Net
 */
TestResult testIntrinsicsUNet() {
    Timer timer;
    TestResult result{"Intrinsics-Conditioned U-Net", false, "", 0.0};

    try {
        auto model = IntrinsicsConditionedUNet(3, 64, 4, 10.0f);

        auto input = torch::randn({2, 3, 256, 256});
        auto intrinsics = torch::tensor({{500.0f, 500.0f, 128.0f, 128.0f},
                                        {600.0f, 600.0f, 128.0f, 128.0f}});

        auto output = model(input, intrinsics);

        // Check output shape
        if (output.size(0) != 2 || output.size(1) != 1 ||
            output.size(2) != 256 || output.size(3) != 256) {
            result.message = "Output shape incorrect";
            return result;
        }

        // Check that different intrinsics produce different outputs
        auto intrinsics2 = torch::tensor({{400.0f, 400.0f, 128.0f, 128.0f},
                                         {400.0f, 400.0f, 128.0f, 128.0f}});
        auto output2 = model(input, intrinsics2);

        auto diff = torch::abs(output - output2).mean().item<float>();
        if (diff < 1e-4) {
            result.message = "Output not conditioned on intrinsics";
            return result;
        }

        result.passed = true;
        result.message = std::to_string(model->count_parameters()) + " parameters";

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Geometry-Aware Network
 */
TestResult testGeometryAwareNetwork() {
    Timer timer;
    TestResult result{"Geometry-Aware Network", false, "", 0.0};

    try {
        auto model = GeometryAwareNetwork(3, 32, 4, 10.0f, true, true);

        auto rgb = torch::randn({1, 3, 128, 128});
        auto rays = torch::randn({1, 3, 128, 128});
        auto intrinsics = torch::tensor({{500.0f, 500.0f, 64.0f, 64.0f}});

        auto output = model(rgb, rays, intrinsics);

        // Check output shape
        if (output.size(0) != 1 || output.size(1) != 1 ||
            output.size(2) != 128 || output.size(3) != 128) {
            result.message = "Output shape incorrect";
            return result;
        }

        // Check parameter count
        auto param_count = model->count_parameters();

        // Check memory estimate
        auto memory_mb = model->estimate_memory_mb(1, 128, 128);

        result.passed = true;
        result.message = std::to_string(param_count) + " params, " +
                        std::to_string(static_cast<int>(memory_mb)) + " MB";

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Lightweight Geometry Network
 */
TestResult testLightweightGeometryNetwork() {
    Timer timer;
    TestResult result{"Lightweight Geometry Network", false, "", 0.0};

    try {
        auto model = LightweightGeometryNetwork(3, 32, 4, 10.0f);

        auto rgb = torch::randn({2, 3, 128, 128});
        auto rays = torch::randn({2, 3, 128, 128});
        auto intrinsics = torch::randn({2, 4});

        auto output = model(rgb, rays, intrinsics);

        if (output.sizes() != torch::IntArrayRef({2, 1, 128, 128})) {
            result.message = "Output shape incorrect";
            return result;
        }

        result.passed = true;

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Scale-Invariant Loss
 */
TestResult testScaleInvariantLoss() {
    Timer timer;
    TestResult result{"Scale-Invariant Loss", false, "", 0.0};

    try {
        ScaleInvariantLoss loss(0.5f);

        auto pred = torch::randn({2, 1, 32, 32}).abs() + 0.1f;
        auto gt = torch::randn({2, 1, 32, 32}).abs() + 0.1f;

        auto loss_val = loss.forward(pred, gt);

        // Check scalar output
        if (loss_val.dim() != 1 || loss_val.size(0) != 1) {
            result.message = "Loss should be scalar";
            return result;
        }

        // Check non-negative
        if (loss_val.item<float>() < 0.0f) {
            result.message = "Loss should be non-negative";
            return result;
        }

        // Test with mask
        auto mask = torch::ones({2, 1, 32, 32}).to(torch::kBool);
        auto loss_masked = loss.forward(pred, gt, mask);

        if (loss_masked.item<float>() < 0.0f) {
            result.message = "Masked loss should be non-negative";
            return result;
        }

        result.passed = true;

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Gradient Matching Loss
 */
TestResult testGradientMatchingLoss() {
    Timer timer;
    TestResult result{"Gradient Matching Loss", false, "", 0.0};

    try {
        GradientMatchingLoss loss(4);

        auto pred = torch::randn({2, 1, 64, 64}).abs() + 0.1f;
        auto gt = torch::randn({2, 1, 64, 64}).abs() + 0.1f;

        auto loss_val = loss.forward(pred, gt);

        if (loss_val.dim() != 1 || loss_val.item<float>() < 0.0f) {
            result.message = "Invalid loss value";
            return result;
        }

        result.passed = true;

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Smoothness Loss
 */
TestResult testSmoothnessLoss() {
    Timer timer;
    TestResult result{"Smoothness Loss", false, "", 0.0};

    try {
        SmoothnessLoss loss;

        auto depth = torch::randn({2, 1, 64, 64}).abs() + 0.1f;
        auto image = torch::randn({2, 3, 64, 64});

        auto loss_val = loss.forward(depth, image);

        if (loss_val.dim() != 1 || loss_val.item<float>() < 0.0f) {
            result.message = "Invalid loss value";
            return result;
        }

        result.passed = true;

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Test Combined Depth Loss
 */
TestResult testCombinedDepthLoss() {
    Timer timer;
    TestResult result{"Combined Depth Loss", false, "", 0.0};

    try {
        CombinedDepthLoss loss(1.0f, 0.1f, 0.001f);

        auto pred = torch::randn({2, 1, 64, 64}).abs() + 0.1f;
        auto gt = torch::randn({2, 1, 64, 64}).abs() + 0.1f;
        auto image = torch::randn({2, 3, 64, 64});

        auto loss_val = loss.forward(pred, gt, image);

        if (loss_val.dim() != 1 || loss_val.item<float>() < 0.0f) {
            result.message = "Invalid loss value";
            return result;
        }

        // Test getComponents
        auto components = loss.getComponents(pred, gt, image);

        if (components.find("si_loss") == components.end() ||
            components.find("grad_loss") == components.end() ||
            components.find("smooth_loss") == components.end()) {
            result.message = "Missing loss components";
            return result;
        }

        result.passed = true;
        result.message = "SI=" + std::to_string(components["si_loss"]) +
                        ", Grad=" + std::to_string(components["grad_loss"]);

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Backward pass test for gradient flow
 */
TestResult testBackwardPass() {
    Timer timer;
    TestResult result{"Backward Pass (Gradient Flow)", false, "", 0.0};

    try {
        auto model = GeometryAwareNetwork(3, 16, 4, 10.0f, false, false);
        CombinedDepthLoss loss;

        auto rgb = torch::randn({1, 3, 64, 64}, torch::requires_grad(true));
        auto rays = torch::randn({1, 3, 64, 64});
        auto intrinsics = torch::randn({1, 4});
        auto gt_depth = torch::randn({1, 1, 64, 64}).abs() + 0.1f;
        auto image = torch::randn({1, 3, 64, 64});

        // Forward pass
        auto pred_depth = model(rgb, rays, intrinsics);
        auto loss_val = loss.forward(pred_depth, gt_depth, image);

        // Backward pass
        loss_val.backward();

        // Check gradients exist
        bool has_gradients = false;
        for (const auto& param : model->parameters()) {
            if (param.grad().defined() && param.grad().abs().sum().item<float>() > 0) {
                has_gradients = true;
                break;
            }
        }

        if (!has_gradients) {
            result.message = "No gradients computed";
            return result;
        }

        result.passed = true;
        result.message = "Gradients flow correctly";

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }

    result.duration_ms = timer.elapsed_ms();
    return result;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Camera-Aware Depth Estimation Tests  \n";
    std::cout << "========================================\n\n";

    std::vector<TestResult> results;

    std::cout << "Layer Tests:\n";
    std::cout << "------------\n";
    results.push_back(testFiLMLayer());
    printTestResult(results.back());

    results.push_back(testCBAM());
    printTestResult(results.back());

    results.push_back(testPCL());
    printTestResult(results.back());

    std::cout << "\nModel Tests:\n";
    std::cout << "------------\n";
    results.push_back(testBaselineUNet());
    printTestResult(results.back());

    results.push_back(testIntrinsicsUNet());
    printTestResult(results.back());

    results.push_back(testGeometryAwareNetwork());
    printTestResult(results.back());

    results.push_back(testLightweightGeometryNetwork());
    printTestResult(results.back());

    std::cout << "\nLoss Function Tests:\n";
    std::cout << "--------------------\n";
    results.push_back(testScaleInvariantLoss());
    printTestResult(results.back());

    results.push_back(testGradientMatchingLoss());
    printTestResult(results.back());

    results.push_back(testSmoothnessLoss());
    printTestResult(results.back());

    results.push_back(testCombinedDepthLoss());
    printTestResult(results.back());

    std::cout << "\nGradient Tests:\n";
    std::cout << "---------------\n";
    results.push_back(testBackwardPass());
    printTestResult(results.back());

    // Summary
    int passed = 0;
    int failed = 0;
    double total_time = 0.0;

    for (const auto& r : results) {
        if (r.passed) passed++;
        else failed++;
        total_time += r.duration_ms;
    }

    std::cout << "\n========================================\n";
    std::cout << "Summary:\n";
    std::cout << "  Total tests: " << results.size() << "\n";
    std::cout << "  \033[32mPassed: " << passed << "\033[0m\n";
    if (failed > 0) {
        std::cout << "  \033[31mFailed: " << failed << "\033[0m\n";
    }
    std::cout << "  Total time: " << std::fixed << std::setprecision(2)
              << total_time << " ms\n";
    std::cout << "========================================\n\n";

    return failed > 0 ? 1 : 0;
}
