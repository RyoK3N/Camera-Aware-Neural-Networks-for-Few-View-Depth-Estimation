/**
 * Main entry point for model evaluation on test set
 *
 * Features:
 * - Comprehensive metric computation on test set
 * - Qualitative visualization generation
 * - Statistical analysis and confidence intervals
 * - Comparison against multiple checkpoints
 * - Export results to CSV/JSON for analysis
 *
 * Usage:
 *   ./evaluate --checkpoint path/to/model.pt --config config.yaml --output results/
 */

#include <torch/torch.h>
#include <cxxopts.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <chrono>

#include "evaluator.h"
#include "../visualization/depth_visualizer.h"
#include "statistical_tests.h"
#include "../data/sunrgbd_loader.h"
#include "../models/baseline_unet.h"
#include "../models/intrinsics_unet.h"
#include "../models/geometry_aware_network.h"

namespace fs = std::filesystem;

/**
 * Configuration for evaluation
 */
struct EvalConfig {
    std::string checkpoint_path;
    std::string config_path;
    std::string data_dir;
    std::string output_dir;
    std::string model_type;  // "baseline", "intrinsics", "geometry"
    std::string vis_colormap = "viridis";

    int batch_size = 1;  // Use 1 for evaluation to get per-sample metrics
    int num_workers = 4;
    int num_vis_samples = 50;  // Number of samples to visualize

    float min_depth = 0.1f;
    float max_depth = 10.0f;

    bool save_all_predictions = false;
    bool compute_confidence_intervals = true;
    bool generate_visualizations = true;
    bool verbose = true;

    static EvalConfig fromYAML(const std::string& yaml_path) {
        YAML::Node config = YAML::LoadFile(yaml_path);
        EvalConfig eval_config;

        // Model parameters
        eval_config.model_type = config["model"]["type"].as<std::string>();

        // Data parameters
        eval_config.data_dir = config["data"]["data_dir"].as<std::string>();
        eval_config.batch_size = config["data"]["batch_size"].as<int>(1);
        eval_config.num_workers = config["data"]["num_workers"].as<int>(4);

        // Depth parameters
        eval_config.min_depth = config["training"]["min_depth"].as<float>(0.1f);
        eval_config.max_depth = config["training"]["max_depth"].as<float>(10.0f);

        return eval_config;
    }
};

/**
 * Load trained model from checkpoint
 */
torch::nn::AnyModule loadModel(const EvalConfig& config, torch::Device device) {
    std::cout << "Loading model from: " << config.checkpoint_path << std::endl;

    torch::nn::AnyModule model(nullptr);

    if (config.model_type == "baseline") {
        auto net = BaselineUNet(/*in_channels=*/3, /*max_depth=*/config.max_depth);
        model = torch::nn::AnyModule(net);
    } else if (config.model_type == "intrinsics") {
        auto net = IntrinsicsConditionedUNet(
            /*in_channels=*/3,
            /*max_depth=*/config.max_depth,
            /*intrinsics_dim=*/4
        );
        model = torch::nn::AnyModule(net);
    } else if (config.model_type == "intrinsics_attention") {
        auto net = IntrinsicsAttentionUNet(
            /*in_channels=*/3,
            /*max_depth=*/config.max_depth,
            /*intrinsics_dim=*/4
        );
        model = torch::nn::AnyModule(net);
    } else if (config.model_type == "geometry_aware") {
        auto net = GeometryAwareNetwork(
            /*in_channels=*/3,
            /*max_depth=*/config.max_depth,
            /*use_pcl=*/true,
            /*use_cbam=*/true
        );
        model = torch::nn::AnyModule(net);
    } else if (config.model_type == "lightweight_geometry") {
        auto net = LightweightGeometryNetwork(
            /*in_channels=*/3,
            /*max_depth=*/config.max_depth
        );
        model = torch::nn::AnyModule(net);
    } else {
        throw std::runtime_error("Unknown model type: " + config.model_type);
    }

    // Load checkpoint
    torch::load(model, config.checkpoint_path);
    model.ptr()->to(device);
    model.ptr()->eval();

    std::cout << "Model loaded successfully" << std::endl;
    return model;
}

/**
 * Create data loader for evaluation
 */
std::unique_ptr<torch::data::DataLoader<>> createDataLoader(const EvalConfig& config) {
    std::cout << "Creating test data loader..." << std::endl;

    auto dataset = SUNRGBDDataset(
        config.data_dir,
        /*split=*/"test",
        /*image_size=*/std::pair<int, int>(480, 640),
        /*load_ray_directions=*/true
    );

    std::cout << "Test set size: " << dataset.size().value() << " samples" << std::endl;

    auto loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(config.batch_size)
            .workers(config.num_workers)
    );

    return loader;
}

/**
 * Generate qualitative visualizations
 */
void generateVisualizations(
    const std::vector<SampleEvaluationResult>& sample_results,
    const EvalConfig& config,
    const std::string& output_dir
) {
    if (!config.generate_visualizations) {
        return;
    }

    std::cout << "\nGenerating visualizations..." << std::endl;
    fs::create_directories(output_dir);

    // Determine colormap
    DepthVisualizer::ColormapType colormap = DepthVisualizer::ColormapType::VIRIDIS;
    if (config.vis_colormap == "plasma") {
        colormap = DepthVisualizer::ColormapType::PLASMA;
    } else if (config.vis_colormap == "magma") {
        colormap = DepthVisualizer::ColormapType::MAGMA;
    } else if (config.vis_colormap == "inferno") {
        colormap = DepthVisualizer::ColormapType::INFERNO;
    } else if (config.vis_colormap == "turbo") {
        colormap = DepthVisualizer::ColormapType::TURBO;
    }

    // Select samples to visualize (uniform sampling)
    int num_vis = std::min(config.num_vis_samples, static_cast<int>(sample_results.size()));
    int step = sample_results.size() / num_vis;

    for (int i = 0; i < num_vis; ++i) {
        int idx = i * step;
        const auto& result = sample_results[idx];

        // Create comprehensive visualization
        auto vis = DepthVisualizer::createComprehensiveVisualization(
            result.rgb_image,
            result.pred_depth,
            result.gt_depth,
            result.metrics,
            config.min_depth,
            config.max_depth,
            colormap
        );

        // Save visualization
        std::string filename = "vis_" + std::to_string(result.sample_idx) + ".png";
        cv::imwrite(output_dir + "/" + filename, vis);

        if ((i + 1) % 10 == 0) {
            std::cout << "  Generated " << (i + 1) << "/" << num_vis << " visualizations" << std::endl;
        }
    }

    std::cout << "Visualizations saved to: " << output_dir << std::endl;
}

/**
 * Generate evaluation report
 */
void generateReport(
    const EvaluationResult& eval_result,
    const EvalConfig& config,
    const std::string& output_dir
) {
    std::string report_path = output_dir + "/evaluation_report.txt";
    std::ofstream report(report_path);

    if (!report.is_open()) {
        std::cerr << "Failed to create report file: " << report_path << std::endl;
        return;
    }

    report << "=================================================================\n";
    report << "                  Model Evaluation Report\n";
    report << "=================================================================\n\n";

    report << "Checkpoint: " << config.checkpoint_path << "\n";
    report << "Model Type: " << config.model_type << "\n";
    report << "Test Samples: " << eval_result.sample_results.size() << "\n";
    report << "Evaluation Date: " << std::put_time(std::localtime(&std::time(nullptr)), "%Y-%m-%d %H:%M:%S") << "\n\n";

    report << "-----------------------------------------------------------------\n";
    report << "                    Performance Metrics\n";
    report << "-----------------------------------------------------------------\n\n";

    // Mean metrics
    report << std::fixed << std::setprecision(4);
    report << "Lower is better:\n";
    report << "  AbsRel:   " << eval_result.mean_metrics.at("abs_rel") << " ± " << eval_result.std_metrics.at("abs_rel") << "\n";
    report << "  SqRel:    " << eval_result.mean_metrics.at("sq_rel") << " ± " << eval_result.std_metrics.at("sq_rel") << "\n";
    report << "  RMSE:     " << eval_result.mean_metrics.at("rmse") << " ± " << eval_result.std_metrics.at("rmse") << "\n";
    report << "  RMSElog:  " << eval_result.mean_metrics.at("rmse_log") << " ± " << eval_result.std_metrics.at("rmse_log") << "\n";
    report << "  MAE:      " << eval_result.mean_metrics.at("mae") << " ± " << eval_result.std_metrics.at("mae") << "\n";
    report << "  Log10:    " << eval_result.mean_metrics.at("log10") << " ± " << eval_result.std_metrics.at("log10") << "\n\n";

    report << "Higher is better (Accuracy thresholds):\n";
    report << "  δ < 1.25:    " << eval_result.mean_metrics.at("delta1") << " ± " << eval_result.std_metrics.at("delta1") << "\n";
    report << "  δ < 1.25²:   " << eval_result.mean_metrics.at("delta2") << " ± " << eval_result.std_metrics.at("delta2") << "\n";
    report << "  δ < 1.25³:   " << eval_result.mean_metrics.at("delta3") << " ± " << eval_result.std_metrics.at("delta3") << "\n\n";

    report << "-----------------------------------------------------------------\n";
    report << "                   Inference Performance\n";
    report << "-----------------------------------------------------------------\n\n";

    report << "  Mean Time:   " << eval_result.mean_inference_time_ms << " ms\n";
    report << "  Std Time:    " << eval_result.std_inference_time_ms << " ms\n";
    report << "  Min Time:    " << eval_result.min_inference_time_ms << " ms\n";
    report << "  Max Time:    " << eval_result.max_inference_time_ms << " ms\n";
    report << "  FPS:         " << (1000.0f / eval_result.mean_inference_time_ms) << "\n\n";

    report << "-----------------------------------------------------------------\n";
    report << "                     Median Metrics\n";
    report << "-----------------------------------------------------------------\n\n";

    report << "  AbsRel:   " << eval_result.median_metrics.at("abs_rel") << "\n";
    report << "  RMSE:     " << eval_result.median_metrics.at("rmse") << "\n";
    report << "  RMSElog:  " << eval_result.median_metrics.at("rmse_log") << "\n";
    report << "  δ < 1.25: " << eval_result.median_metrics.at("delta1") << "\n\n";

    report << "=================================================================\n";

    report.close();
    std::cout << "Evaluation report saved to: " << report_path << std::endl;
}

/**
 * Main evaluation function
 */
int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        cxxopts::Options options("evaluate", "Evaluate trained depth estimation model on test set");

        options.add_options()
            ("c,checkpoint", "Path to model checkpoint", cxxopts::value<std::string>())
            ("g,config", "Path to config YAML", cxxopts::value<std::string>())
            ("o,output", "Output directory for results", cxxopts::value<std::string>()->default_value("results/eval"))
            ("data-dir", "Data directory (overrides config)", cxxopts::value<std::string>()->default_value(""))
            ("num-vis", "Number of visualizations to generate", cxxopts::value<int>()->default_value("50"))
            ("colormap", "Colormap for visualization", cxxopts::value<std::string>()->default_value("viridis"))
            ("no-vis", "Disable visualization generation")
            ("save-predictions", "Save all depth predictions")
            ("h,help", "Print usage");

        auto args = options.parse(argc, argv);

        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        if (!args.count("checkpoint") || !args.count("config")) {
            std::cerr << "Error: --checkpoint and --config are required" << std::endl;
            std::cout << options.help() << std::endl;
            return 1;
        }

        // Load configuration
        EvalConfig config = EvalConfig::fromYAML(args["config"].as<std::string>());
        config.checkpoint_path = args["checkpoint"].as<std::string>();
        config.config_path = args["config"].as<std::string>();
        config.output_dir = args["output"].as<std::string>();
        config.num_vis_samples = args["num-vis"].as<int>();
        config.vis_colormap = args["colormap"].as<std::string>();
        config.generate_visualizations = !args.count("no-vis");
        config.save_all_predictions = args.count("save-predictions");

        if (args["data-dir"].as<std::string>() != "") {
            config.data_dir = args["data-dir"].as<std::string>();
        }

        // Verify checkpoint exists
        if (!fs::exists(config.checkpoint_path)) {
            std::cerr << "Error: Checkpoint not found: " << config.checkpoint_path << std::endl;
            return 1;
        }

        // Create output directory
        fs::create_directories(config.output_dir);

        // Set device
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "CUDA not available, using CPU" << std::endl;
        }

        std::cout << "\n=================================================================\n";
        std::cout << "                    Starting Evaluation\n";
        std::cout << "=================================================================\n\n";

        // Load model
        auto model = loadModel(config, device);

        // Create data loader
        auto test_loader = createDataLoader(config);

        // Create evaluator
        ModelEvaluator evaluator(
            model,
            device,
            config.min_depth,
            config.max_depth,
            config.verbose
        );

        // Run evaluation
        std::cout << "\nRunning evaluation on test set...\n" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        EvaluationResult eval_result = evaluator.evaluate(*test_loader);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "\n=================================================================\n";
        std::cout << "                   Evaluation Complete\n";
        std::cout << "=================================================================\n\n";
        std::cout << "Total evaluation time: " << duration.count() << " seconds" << std::endl;

        // Save results
        std::cout << "\nSaving results..." << std::endl;
        evaluator.saveResults(eval_result, config.output_dir);

        // Generate visualizations
        if (config.generate_visualizations) {
            std::string vis_dir = config.output_dir + "/visualizations";
            generateVisualizations(eval_result.sample_results, config, vis_dir);
        }

        // Generate report
        generateReport(eval_result, config, config.output_dir);

        // Print summary
        std::cout << "\n=================================================================\n";
        std::cout << "                      Summary Statistics\n";
        std::cout << "=================================================================\n\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "AbsRel:   " << eval_result.mean_metrics.at("abs_rel") << " ± " << eval_result.std_metrics.at("abs_rel") << std::endl;
        std::cout << "RMSE:     " << eval_result.mean_metrics.at("rmse") << " ± " << eval_result.std_metrics.at("rmse") << std::endl;
        std::cout << "δ < 1.25: " << eval_result.mean_metrics.at("delta1") << " ± " << eval_result.std_metrics.at("delta1") << std::endl;
        std::cout << "\nMean inference time: " << eval_result.mean_inference_time_ms << " ms ("
                  << (1000.0f / eval_result.mean_inference_time_ms) << " FPS)" << std::endl;

        std::cout << "\nAll results saved to: " << config.output_dir << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
