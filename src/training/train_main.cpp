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
#include "../training/production_trainer.h"
#include "../training/tensorboard_trainer.h"
#include "../training/tensorboard_trainer_enhanced.h"
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
        ("tensorboard", "Enable TensorBoard logging", cxxopts::value<bool>()->default_value("true"))
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
        train_config.reproj_weight = loss["reproj_weight"].as<float>(0.01f);
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
              << " Smooth=" << config.smooth_weight
              << " Reproj=" << config.reproj_weight << "\n";
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
        bool use_tensorboard = args["tensorboard"].as<bool>();

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

        // For now, create a simple baseline model directly
        // TODO: Use createModel() with proper type casting
        int init_features = 64;
        float max_depth = 10.0f;
        if (yaml_config["model"]) {
            init_features = yaml_config["model"]["init_features"].as<int>(64);
            max_depth = yaml_config["model"]["max_depth"].as<float>(10.0f);
        }

        auto model_impl = std::make_shared<BaselineUNetImpl>(3, init_features, max_depth);
        auto model = std::dynamic_pointer_cast<torch::nn::Module>(model_impl);

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
            config.smooth_weight,
            config.reproj_weight
        );

        // Create data loaders
        std::cout << "Loading dataset...\n";

        // Load data configuration
        std::string data_dir = yaml_config["data"]["data_dir"].as<std::string>("./data/sunrgbd");
        std::string manifest_path = yaml_config["data"]["manifest_path"].as<std::string>("./data/sunrgbd_manifest.json");

        std::cout << "  Data directory: " << data_dir << "\n";
        std::cout << "  Manifest path: " << manifest_path << "\n";

        // Create training data loader
        auto train_loader_ptr = std::make_shared<SunRGBDLoader>(
            data_dir,
            manifest_path,
            "train"
        );

        // Enable augmentation for training
        AugmentationConfig aug_config;
        aug_config.enable_random_crop = yaml_config["data"]["augmentation"]["random_crop"].as<bool>(true);
        aug_config.enable_horizontal_flip = yaml_config["data"]["augmentation"]["horizontal_flip"].as<bool>(true);
        aug_config.horizontal_flip_prob = yaml_config["data"]["augmentation"]["flip_probability"].as<float>(0.5f);
        aug_config.enable_color_jitter = yaml_config["data"]["augmentation"]["color_jitter"].as<bool>(true);
        aug_config.brightness_delta = yaml_config["data"]["augmentation"]["brightness"].as<float>(0.2f);
        aug_config.contrast_delta = yaml_config["data"]["augmentation"]["contrast"].as<float>(0.2f);
        aug_config.saturation_delta = yaml_config["data"]["augmentation"]["saturation"].as<float>(0.2f);
        aug_config.hue_delta = yaml_config["data"]["augmentation"]["hue"].as<float>(0.1f);
        train_loader_ptr->enableAugmentation(aug_config);

        // Set target dimensions
        int input_height = yaml_config["data"]["input_height"].as<int>(240);
        int input_width = yaml_config["data"]["input_width"].as<int>(320);
        train_loader_ptr->setTargetDimensions(input_height, input_width);

        // Create validation data loader
        auto val_loader_ptr = std::make_shared<SunRGBDLoader>(
            data_dir,
            manifest_path,
            "test"  // Using test split for validation
        );
        val_loader_ptr->disableAugmentation();
        val_loader_ptr->setTargetDimensions(input_height, input_width);

        std::cout << "Training samples: " << train_loader_ptr->size() << "\n";
        std::cout << "Validation samples: " << val_loader_ptr->size() << "\n";
        std::cout << "\n";

        // Create trainer based on configuration
        if (use_tensorboard) {
            // TensorBoard trainer with state-of-the-art visualization
            TensorBoardTrainerEnhanced::Config tb_config;
            tb_config.num_epochs = config.num_epochs;
            tb_config.batch_size = config.batch_size;
            tb_config.learning_rate = config.learning_rate;
            tb_config.weight_decay = config.weight_decay;
            tb_config.use_grad_clip = config.use_grad_clip;
            tb_config.grad_clip_value = config.grad_clip_value;
            tb_config.val_interval = config.val_interval;
            tb_config.log_interval = config.log_interval;
            tb_config.save_interval = config.save_interval;
            tb_config.checkpoint_dir = config.checkpoint_dir;
            tb_config.log_dir = config.log_dir;
            tb_config.tensorboard_dir = yaml_config["logging"]["tensorboard"]["tensorboard_dir"]
                .as<std::string>("./runs");
            tb_config.experiment_name = config.experiment_name;
            tb_config.device = device;
            tb_config.viz_interval = yaml_config["logging"]["tensorboard"]["log_image_interval"]
                .as<int>(1);
            tb_config.num_viz_samples = yaml_config["visualization"]["num_viz_samples"]
                .as<int>(4);
            tb_config.histogram_interval = yaml_config["logging"]["tensorboard"]["log_histogram_interval"]
                .as<int>(5);

            std::cout << "Initializing TensorBoard trainer (Enhanced)...\n";
            std::cout << "  Batch size: " << tb_config.batch_size << "\n";
            std::cout << "  Learning rate: " << tb_config.learning_rate << "\n";
            std::cout << "  Validation interval: " << tb_config.val_interval << " epochs\n";
            std::cout << "  Logging to: " << tb_config.log_dir << "\n";
            std::cout << "  TensorBoard to: " << tb_config.tensorboard_dir << "\n";
            std::cout << "  Checkpoints to: " << tb_config.checkpoint_dir << "\n";
            std::cout << "\n";

            TensorBoardTrainerEnhanced trainer(model, model_impl, loss_fn, tb_config);

            std::cout << "════════════════════════════════════════════════════════════\n";
            std::cout << "Starting training with TensorBoard...\n";
            std::cout << "════════════════════════════════════════════════════════════\n";

            trainer.train(train_loader_ptr, val_loader_ptr);

            std::cout << "\n";
            std::cout << "════════════════════════════════════════════════════════════\n";
            std::cout << "Training completed successfully!\n";
            std::cout << "Checkpoints saved in: " << tb_config.checkpoint_dir << "\n";
            std::cout << "Logs saved in: " << tb_config.log_dir << "\n";
            std::cout << "TensorBoard logs in: " << tb_config.tensorboard_dir << "\n";
            std::cout << "\nView TensorBoard:\n";
            std::cout << "  tensorboard --logdir=" << tb_config.tensorboard_dir << "\n";
            std::cout << "  Then open: http://localhost:6006\n";
            std::cout << "\n";
        } else {
            // Production trainer without TensorBoard
            ProductionTrainer::Config trainer_config;
            trainer_config.num_epochs = config.num_epochs;
            trainer_config.batch_size = config.batch_size;
            trainer_config.learning_rate = config.learning_rate;
            trainer_config.weight_decay = config.weight_decay;
            trainer_config.use_grad_clip = config.use_grad_clip;
            trainer_config.grad_clip_value = config.grad_clip_value;
            trainer_config.val_interval = config.val_interval;
            trainer_config.log_interval = config.log_interval;
            trainer_config.save_interval = config.save_interval;
            trainer_config.checkpoint_dir = config.checkpoint_dir;
            trainer_config.log_dir = config.log_dir;
            trainer_config.experiment_name = config.experiment_name;
            trainer_config.device = device;

            std::cout << "Initializing production trainer...\n";
            std::cout << "  Batch size: " << trainer_config.batch_size << "\n";
            std::cout << "  Learning rate: " << trainer_config.learning_rate << "\n";
            std::cout << "  Logging to: " << trainer_config.log_dir << "\n";
            std::cout << "  Checkpoints to: " << trainer_config.checkpoint_dir << "\n";
            std::cout << "\n";

            ProductionTrainer trainer(model, model_impl, loss_fn, trainer_config);

            std::cout << "════════════════════════════════════════════════════════════\n";
            std::cout << "Starting training...\n";
            std::cout << "════════════════════════════════════════════════════════════\n";

            trainer.train(train_loader_ptr, val_loader_ptr);

            std::cout << "\n";
            std::cout << "════════════════════════════════════════════════════════════\n";
            std::cout << "Training completed successfully!\n";
            std::cout << "Checkpoints saved in: " << config.checkpoint_dir << "\n";
            std::cout << "Logs saved in: " << config.log_dir << "\n";
            std::cout << "  - training.log: Timestamped training log\n";
            std::cout << "  - metrics.csv: Epoch-by-epoch metrics\n";
            std::cout << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
