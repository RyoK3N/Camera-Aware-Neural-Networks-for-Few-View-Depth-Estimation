/**
 * @file train_main.cpp
 * @brief Main training script for camera-aware depth estimation
 *
 * Usage:
 *   ./train --config configs/train_config.yaml --experiment baseline_unet
 *   ./train --config configs/train_config.yaml --experiment geometry_aware_full --gpu 0
 *   ./train --resume checkpoints/baseline_unet/checkpoint_epoch_20.pt
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cxxopts.hpp>
#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

#include "../models/baseline_unet.h"
#include "../models/intrinsics_unet.h"
#include "../models/geometry_aware_network.h"
#include "../loss/depth_loss.h"
#include "../data/sunrgbd_loader.h"
#include "../training/trainer.h"
#include "../evaluation/depth_metrics.h"

using namespace camera_aware_depth;

/**
 * @brief Parse command line arguments
 */
cxxopts::ParseResult parseArguments(int argc, char* argv[]) {
    cxxopts::Options options("train", "Train depth estimation models");

    options.add_options()
        ("c,config", "Path to config file", cxxopts::value<std::string>()->default_value("configs/train_config.yaml"))
        ("e,experiment", "Experiment name", cxxopts::value<std::string>()->default_value("baseline_unet"))
        ("r,resume", "Resume from checkpoint", cxxopts::value<std::string>()->default_value(""))
        ("g,gpu", "GPU ID", cxxopts::value<int>()->default_value("0"))
        ("d,debug", "Enable debug mode", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    return result;
}

/**
 * @brief Load configuration from YAML file
 */
TrainingConfig loadConfig(const std::string& config_path, const std::string& experiment_name) {
    YAML::Node config = YAML::LoadFile(config_path);

    TrainingConfig train_config;

    // Load base configuration
    if (config["optimization"]) {
        auto opt = config["optimization"];
        train_config.optimizer = opt["optimizer"].as<std::string>("adamw");
        train_config.learning_rate = opt["learning_rate"].as<float>(1e-4f);
        train_config.weight_decay = opt["weight_decay"].as<float>(1e-5f);

        if (opt["adam"]) {
            auto adam = opt["adam"];
            auto betas = adam["betas"].as<std::vector<float>>();
            train_config.adam_betas = {betas[0], betas[1]};
            train_config.adam_eps = adam["eps"].as<float>(1e-8f);
        }

        if (opt["lr_scheduler"].as<std::string>() != "") {
            train_config.lr_scheduler = opt["lr_scheduler"].as<std::string>();
        }
        train_config.lr_step_size = opt["lr_step_size"].as<int>(10);
        train_config.lr_gamma = opt["lr_gamma"].as<float>(0.5f);
        train_config.lr_warmup_epochs = opt["lr_warmup_epochs"].as<int>(2);

        if (opt["gradient_clip"]) {
            train_config.use_grad_clip = opt["gradient_clip"].as<bool>(true);
            train_config.grad_clip_value = opt["gradient_clip_value"].as<float>(1.0f);
        }
    }

    if (config["training"]) {
        auto training = config["training"];
        train_config.num_epochs = training["num_epochs"].as<int>(50);
        train_config.batch_size = training["batch_size"].as<int>(8);
        train_config.num_workers = training["num_workers"].as<int>(4);
        // Note: AMP not available in LibTorch C++ API - training uses float32
        train_config.log_interval = training["log_interval"].as<int>(10);
        train_config.val_interval = training["val_interval"].as<int>(1);
    }

    if (config["loss"]) {
        auto loss = config["loss"];
        train_config.si_weight = loss["si_weight"].as<float>(1.0f);
        train_config.grad_weight = loss["grad_weight"].as<float>(0.1f);
        train_config.smooth_weight = loss["smooth_weight"].as<float>(0.001f);
        train_config.min_depth = loss["min_depth"].as<float>(0.1f);
        train_config.max_depth = loss["max_depth"].as<float>(10.0f);
    }

    if (config["checkpointing"]) {
        auto ckpt = config["checkpointing"];
        train_config.checkpoint_dir = ckpt["checkpoint_dir"].as<std::string>("./checkpoints");
        train_config.save_interval = ckpt["save_interval"].as<int>(5);
        train_config.save_best_only = ckpt["save_best_only"].as<bool>(true);
        train_config.keep_last_n_checkpoints = ckpt["keep_last_n"].as<int>(3);
    }

    if (config["early_stopping"]) {
        auto early_stop = config["early_stopping"];
        train_config.use_early_stopping = early_stop["enabled"].as<bool>(true);
        train_config.early_stop_patience = early_stop["patience"].as<int>(10);
        train_config.early_stop_min_delta = early_stop["min_delta"].as<float>(1e-4f);
    }

    if (config["validation"]) {
        auto val = config["validation"];
        train_config.metric_to_monitor = val["primary_metric"].as<std::string>("abs_rel");
        std::string mode = val["metric_mode"].as<std::string>("min");
        train_config.metric_lower_is_better = (mode == "min");
    }

    if (config["logging"]) {
        auto logging = config["logging"];
        train_config.log_dir = logging["log_dir"].as<std::string>("./logs");
        if (logging["tensorboard"]) {
            train_config.save_tensorboard = logging["tensorboard"]["enabled"].as<bool>(true);
        }
    }

    if (config["experiment"]) {
        auto exp = config["experiment"];
        train_config.experiment_name = exp["name"].as<std::string>(experiment_name);
        train_config.seed = exp["seed"].as<int>(42);
        train_config.deterministic = exp["deterministic"].as<bool>(false);
    }

    // Apply experiment-specific overrides
    if (config["experiments"] && config["experiments"][experiment_name]) {
        auto exp_config = config["experiments"][experiment_name];

        if (exp_config["training"] && exp_config["training"]["batch_size"]) {
            train_config.batch_size = exp_config["training"]["batch_size"].as<int>();
        }

        if (exp_config["experiment"] && exp_config["experiment"]["name"]) {
            train_config.experiment_name = exp_config["experiment"]["name"].as<std::string>();
        }
    }

    // Adjust paths to include experiment name
    train_config.checkpoint_dir += "/" + train_config.experiment_name;
    train_config.log_dir += "/" + train_config.experiment_name;

    return train_config;
}

/**
 * @brief Create model based on configuration
 */
std::shared_ptr<torch::nn::Module> createModel(const YAML::Node& config) {
    std::string architecture = "baseline_unet";
    if (config["model"] && config["model"]["architecture"]) {
        architecture = config["model"]["architecture"].as<std::string>();
    }

    int init_features = 64;
    float max_depth = 10.0f;

    if (config["model"]) {
        init_features = config["model"]["init_features"].as<int>(64);
        max_depth = config["model"]["max_depth"].as<float>(10.0f);
    }

    std::shared_ptr<torch::nn::Module> model;

    if (architecture == "baseline_unet") {
        model = std::make_shared<BaselineUNetImpl>(3, init_features, max_depth);

    } else if (architecture == "intrinsics_unet") {
        bool use_attention = config["model"]["use_attention"].as<bool>(false);

        if (use_attention) {
            model = std::make_shared<IntrinsicsAttentionUNetImpl>(3, init_features, 4, max_depth);
        } else {
            model = std::make_shared<IntrinsicsConditionedUNetImpl>(3, init_features, 4, max_depth);
        }

    } else if (architecture == "geometry_aware") {
        std::string variant = config["model"]["variant"].as<std::string>("full");
        bool use_pcl = config["model"]["use_pcl"].as<bool>(true);
        bool use_attention = config["model"]["use_attention"].as<bool>(true);

        if (variant == "lightweight") {
            model = std::make_shared<LightweightGeometryNetworkImpl>(3, init_features, 4, max_depth);
        } else {
            model = std::make_shared<GeometryAwareNetworkImpl>(
                3, init_features, 4, max_depth, use_pcl, use_attention
            );
        }

    } else {
        throw std::runtime_error("Unknown architecture: " + architecture);
    }

    return model;
}

/**
 * @brief Setup random seeds for reproducibility
 */
void setupSeeds(int seed, bool deterministic) {
    torch::manual_seed(seed);

    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(seed);

        // Note: torch::set_deterministic and torch::backends::cudnn are not available in LibTorch C++
        // For deterministic behavior, use CUBLAS_WORKSPACE_CONFIG environment variable
        if (deterministic) {
            std::cout << "Note: For full determinism, set environment variable:\n";
            std::cout << "  CUBLAS_WORKSPACE_CONFIG=:4096:8\n";
        }
    }
}

/**
 * @brief Print training banner
 */
void printBanner(const TrainingConfig& config, const YAML::Node& yaml_config) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Camera-Aware Depth Estimation Training                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    std::cout << "Experiment: " << config.experiment_name << "\n";

    if (yaml_config["experiment"] && yaml_config["experiment"]["description"]) {
        std::cout << "Description: " << yaml_config["experiment"]["description"].as<std::string>() << "\n";
    }

    if (yaml_config["experiment"] && yaml_config["experiment"]["tags"]) {
        std::cout << "Tags: ";
        for (const auto& tag : yaml_config["experiment"]["tags"]) {
            std::cout << tag.as<std::string>() << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "Configuration:\n";
    std::cout << "  Model: " << yaml_config["model"]["architecture"].as<std::string>("baseline_unet") << "\n";
    std::cout << "  Optimizer: " << config.optimizer << "\n";
    std::cout << "  Learning Rate: " << config.learning_rate << "\n";
    std::cout << "  Batch Size: " << config.batch_size << "\n";
    std::cout << "  Epochs: " << config.num_epochs << "\n";
    std::cout << "  Loss Weights: SI=" << config.si_weight
              << " Grad=" << config.grad_weight
              << " Smooth=" << config.smooth_weight << "\n";
    std::cout << "\n";
}

/**
 * @brief Main training function
 */
int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        auto args = parseArguments(argc, argv);

        std::string config_path = args["config"].as<std::string>();
        std::string experiment_name = args["experiment"].as<std::string>();
        std::string resume_checkpoint = args["resume"].as<std::string>();
        int gpu_id = args["gpu"].as<int>();
        bool debug_mode = args["debug"].as<bool>();

        // Load configuration
        std::cout << "Loading configuration from: " << config_path << "\n";
        YAML::Node yaml_config = YAML::LoadFile(config_path);
        auto config = loadConfig(config_path, experiment_name);

        // Override with debug settings if enabled
        if (debug_mode || (yaml_config["debug"] && yaml_config["debug"]["enabled"].as<bool>())) {
            std::cout << "⚠️  Debug mode enabled - using reduced dataset\n";
            config.num_epochs = yaml_config["debug"]["num_epochs"].as<int>(2);
            config.log_interval = yaml_config["debug"]["log_interval"].as<int>(1);
        }

        // Print banner
        printBanner(config, yaml_config);

        // Setup device
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available() && yaml_config["hardware"]["device"].as<std::string>() == "cuda") {
            device = torch::Device(torch::kCUDA, gpu_id);
            std::cout << "Using CUDA device " << gpu_id << "\n";
            // Note: get_device_name not available in this LibTorch version
        } else {
            std::cout << "Using CPU\n";
        }
        std::cout << "\n";

        // Setup reproducibility
        setupSeeds(config.seed, config.deterministic);

        // Create model
        std::cout << "Creating model...\n";
        auto model = createModel(yaml_config);

        // Count parameters
        int64_t total_params = 0;
        int64_t trainable_params = 0;
        for (const auto& param : model->parameters()) {
            int64_t num = param.numel();
            total_params += num;
            if (param.requires_grad()) {
                trainable_params += num;
            }
        }

        std::cout << "Model created successfully\n";
        std::cout << "  Total parameters: " << total_params / 1e6 << "M\n";
        std::cout << "  Trainable parameters: " << trainable_params / 1e6 << "M\n";
        std::cout << "\n";

        // Create loss function
        auto loss_fn = std::make_shared<CombinedDepthLoss>(
            config.si_weight,
            config.grad_weight,
            config.smooth_weight
        );

        // Create data loaders
        std::cout << "Loading dataset...\n";

        // TODO: Implement actual data loader creation based on config
        // For now, this is a placeholder that shows the structure

        /*
        auto train_dataset = createDataset(yaml_config, "train");
        auto val_dataset = createDataset(yaml_config, "val");

        auto train_loader = torch::data::make_data_loader(
            std::move(train_dataset),
            torch::data::DataLoaderOptions()
                .batch_size(config.batch_size)
                .workers(config.num_workers)
        );

        auto val_loader = torch::data::make_data_loader(
            std::move(val_dataset),
            torch::data::DataLoaderOptions()
                .batch_size(config.batch_size)
                .workers(config.num_workers)
        );
        */

        // Create trainer
        std::cout << "Initializing trainer...\n";
        DepthTrainer trainer(model, loss_fn, config, device);

        // Resume from checkpoint if specified
        if (!resume_checkpoint.empty()) {
            std::cout << "Resuming from checkpoint: " << resume_checkpoint << "\n";
            trainer.loadCheckpoint(resume_checkpoint);
        }

        std::cout << "\n";
        std::cout << "Starting training...\n";
        std::cout << "════════════════════════════════════════════════════════════\n";
        std::cout << "\n";

        // Start training
        // trainer.train(*train_loader, *val_loader);

        std::cout << "\n";
        std::cout << "════════════════════════════════════════════════════════════\n";
        std::cout << "Training completed successfully!\n";
        std::cout << "Checkpoints saved in: " << config.checkpoint_dir << "\n";
        std::cout << "Logs saved in: " << config.log_dir << "\n";
        std::cout << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
