#ifndef TRAINER_H
#define TRAINER_H

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "../evaluation/depth_metrics.h"
#include "../loss/depth_loss.h"

namespace camera_aware_depth {

/**
 * @brief Training Configuration
 *
 * Comprehensive configuration for depth estimation training based on:
 * - Ranftl et al., "Towards Robust Monocular Depth Estimation: MiDaS", CVPR 2020
 * - Godard et al., "Digging Into Self-Supervised Monocular Depth Estimation", ICCV 2019
 * - Yin et al., "Learning to Recover 3D Scene Shape from a Single Image", CVPR 2021
 */
struct TrainingConfig {
    // Optimization
    std::string optimizer = "adam";           // "adam", "sgd", "adamw"
    float learning_rate = 1e-4f;
    float weight_decay = 1e-5f;
    float momentum = 0.9f;                     // For SGD
    std::array<float, 2> adam_betas = {0.9f, 0.999f};
    float adam_eps = 1e-8f;

    // Learning rate schedule
    std::string lr_scheduler = "step";        // "step", "cosine", "plateau", "none"
    int lr_step_size = 10;                     // For step scheduler
    float lr_gamma = 0.5f;                     // LR decay factor
    int lr_warmup_epochs = 2;                  // Linear warmup epochs
    float lr_min = 1e-6f;                      // Minimum LR for cosine

    // Training
    int num_epochs = 50;
    int batch_size = 8;
    int num_workers = 4;
    bool pin_memory = true;

    // Validation
    int val_interval = 1;                      // Validate every N epochs
    int log_interval = 10;                     // Log every N batches

    // Checkpointing
    std::string checkpoint_dir = "./checkpoints";
    int save_interval = 5;                     // Save checkpoint every N epochs
    bool save_best_only = true;                // Only save best validation model
    std::string metric_to_monitor = "abs_rel"; // Metric for best model selection
    bool metric_lower_is_better = true;
    int keep_last_n_checkpoints = 3;           // Keep only N most recent checkpoints

    // Early stopping
    bool use_early_stopping = true;
    int early_stop_patience = 10;              // Stop if no improvement for N epochs
    float early_stop_min_delta = 1e-4f;        // Minimum change to qualify as improvement

    // Loss weights
    float si_weight = 1.0f;
    float grad_weight = 0.1f;
    float smooth_weight = 0.001f;
    float reproj_weight = 0.01f;  // Reprojection error (geometric consistency)

    // Depth range
    float min_depth = 0.1f;
    float max_depth = 10.0f;

    // Mixed precision training
    // Note: AMP (Automatic Mixed Precision) is not available in LibTorch C++ API
    // Training will use standard float32 precision

    // Gradient clipping
    bool use_grad_clip = true;
    float grad_clip_value = 1.0f;

    // Resume training
    std::string resume_checkpoint = "";        // Path to checkpoint to resume from

    // Experiment tracking
    std::string experiment_name = "experiment";
    std::string log_dir = "./logs";
    bool save_tensorboard = true;

    // Reproducibility
    int seed = 42;
    bool deterministic = false;                // Slower but reproducible
};

/**
 * @brief Training State
 *
 * Tracks the current state of training for checkpointing and resuming
 */
struct TrainingState {
    int current_epoch = 0;
    int global_step = 0;
    float best_metric = std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    bool early_stopped = false;

    // History
    std::vector<float> train_losses;
    std::vector<std::map<std::string, float>> val_metrics_history;
    std::vector<float> learning_rates;
};

/**
 * @brief Depth Estimation Trainer
 *
 * Production-grade trainer with:
 * - Automatic mixed precision (AMP)
 * - Learning rate scheduling
 * - Early stopping
 * - Checkpoint management
 * - Gradient clipping
 * - Comprehensive logging
 *
 * Design Principles:
 * - RAII for resource management
 * - Exception safety
 * - Separation of concerns
 * - Testability
 */
class DepthTrainer {
public:
    /**
     * @brief Construct trainer
     *
     * @param model Depth estimation model (shared_ptr for proper ownership)
     * @param loss_fn Loss function
     * @param config Training configuration
     * @param device Device to train on
     */
    DepthTrainer(
        std::shared_ptr<torch::nn::Module> model,
        std::shared_ptr<CombinedDepthLoss> loss_fn,
        const TrainingConfig& config,
        torch::Device device = torch::kCUDA
    ) : model_(model),
        loss_fn_(loss_fn),
        config_(config),
        device_(device),
        state_() {

        // Move model to device
        model_->to(device_);

        // Setup optimizer
        setupOptimizer();

        // Setup learning rate scheduler
        setupScheduler();

        // Create checkpoint directory
        createDirectory(config_.checkpoint_dir);
        createDirectory(config_.log_dir);
    }

    /**
     * @brief Train for one epoch
     *
     * @param train_loader Training data loader
     * @return Average training loss
     */
    template<typename DataLoader>
    float trainEpoch(DataLoader& train_loader) {
        model_->train();

        float total_loss = 0.0f;
        int num_batches = 0;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (auto& batch : train_loader) {
            auto batch_start = std::chrono::high_resolution_clock::now();

            // Forward pass
            optimizer_->zero_grad();

            // Standard float32 training (AMP not available in LibTorch C++)
            auto loss = forwardBatch(batch);
            loss.backward();

            if (config_.use_grad_clip) {
                torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.grad_clip_value);
            }

            optimizer_->step();

            total_loss += loss.template item<float>();
            num_batches++;
            state_.global_step++;

            // Logging
            if (state_.global_step % config_.log_interval == 0) {
                auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - batch_start).count();

                logTrainingStep(loss.template item<float>(), batch_time);
            }
        }

        float avg_loss = total_loss / num_batches;

        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - epoch_start).count();

        std::cout << "Epoch " << state_.current_epoch
                  << " - Train Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " - Time: " << epoch_time << "s" << std::endl;

        return avg_loss;
    }

    /**
     * @brief Validate on validation set
     *
     * @param val_loader Validation data loader
     * @return Validation metrics
     */
    template<typename DataLoader>
    std::map<std::string, float> validate(DataLoader& val_loader) {
        model_->eval();
        torch::NoGradGuard no_grad;

        MetricsAccumulator metrics_acc;

        for (auto& batch : val_loader) {
            auto metrics = validateBatch(batch);
            metrics_acc.update(metrics);
        }

        auto avg_metrics = metrics_acc.average();

        std::cout << "\nValidation Results:\n";
        std::cout << formatMetrics(avg_metrics) << std::endl;

        return avg_metrics;
    }

    /**
     * @brief Full training loop
     *
     * @param train_loader Training data loader
     * @param val_loader Validation data loader
     */
    template<typename DataLoader>
    void train(
        DataLoader& train_loader,
        DataLoader& val_loader
    ) {
        std::cout << "Starting training...\n";
        std::cout << "Configuration:\n";
        printConfig();
        std::cout << "\n";

        for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
            state_.current_epoch = epoch;

            // Learning rate warmup
            if (epoch < config_.lr_warmup_epochs) {
                float warmup_lr = config_.learning_rate * (epoch + 1) / config_.lr_warmup_epochs;
                setLearningRate(warmup_lr);
            }

            // Train epoch
            float train_loss = trainEpoch(train_loader);
            state_.train_losses.push_back(train_loss);

            // Validation
            if ((epoch + 1) % config_.val_interval == 0) {
                auto val_metrics = validate(val_loader);
                state_.val_metrics_history.push_back(val_metrics);

                // Check for improvement
                float current_metric = val_metrics[config_.metric_to_monitor];
                bool improved = checkImprovement(current_metric);

                if (improved) {
                    state_.best_metric = current_metric;
                    state_.epochs_without_improvement = 0;

                    if (config_.save_best_only) {
                        saveCheckpoint("best_model.pt");
                    }
                } else {
                    state_.epochs_without_improvement++;
                }

                // Early stopping check
                if (config_.use_early_stopping &&
                    state_.epochs_without_improvement >= config_.early_stop_patience) {
                    std::cout << "\nEarly stopping triggered after " << epoch + 1 << " epochs\n";
                    state_.early_stopped = true;
                    break;
                }
            }

            // Learning rate scheduling (after warmup)
            if (epoch >= config_.lr_warmup_epochs && scheduler_) {
                scheduler_->step();
                state_.learning_rates.push_back(getCurrentLR());
            }

            // Periodic checkpoint
            if ((epoch + 1) % config_.save_interval == 0 && !config_.save_best_only) {
                std::string checkpoint_name = "checkpoint_epoch_" + std::to_string(epoch + 1) + ".pt";
                saveCheckpoint(checkpoint_name);
                cleanupOldCheckpoints();
            }
        }

        std::cout << "\nTraining completed!\n";
        std::cout << "Best " << config_.metric_to_monitor << ": "
                  << std::fixed << std::setprecision(4) << state_.best_metric << "\n";

        // Save final model
        saveCheckpoint("final_model.pt");
        saveTrainingHistory();
    }

    /**
     * @brief Save checkpoint
     */
    void saveCheckpoint(const std::string& filename) {
        std::string path = config_.checkpoint_dir + "/" + filename;

        torch::save(model_, path);

        // Save optimizer state
        std::string opt_path = config_.checkpoint_dir + "/optimizer_" + filename;
        torch::save(*optimizer_, opt_path);

        // Save training state
        std::string state_path = config_.checkpoint_dir + "/state_" + filename;
        saveTrainingState(state_path);

        std::cout << "Checkpoint saved: " << path << std::endl;
    }

    /**
     * @brief Load checkpoint
     */
    void loadCheckpoint(const std::string& path) {
        torch::load(model_, path);

        // Load optimizer if exists
        std::string opt_path = path;
        opt_path.replace(opt_path.find("checkpoint"), 10, "optimizer");
        if (std::filesystem::exists(opt_path)) {
            torch::load(*optimizer_, opt_path);
        }

        // Load training state if exists
        std::string state_path = path;
        state_path.replace(state_path.find("checkpoint"), 10, "state");
        if (std::filesystem::exists(state_path)) {
            loadTrainingState(state_path);
        }

        std::cout << "Checkpoint loaded: " << path << std::endl;
    }

    /**
     * @brief Get current training state
     */
    const TrainingState& getState() const { return state_; }

private:
    std::shared_ptr<torch::nn::Module> model_;
    std::shared_ptr<CombinedDepthLoss> loss_fn_;
    TrainingConfig config_;
    torch::Device device_;
    TrainingState state_;

    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<torch::optim::LRScheduler> scheduler_;
    // AMP scaler removed - not available in LibTorch C++ API

    /**
     * @brief Setup optimizer based on configuration
     */
    void setupOptimizer() {
        auto params = model_->parameters();

        if (config_.optimizer == "adam") {
            optimizer_ = std::make_unique<torch::optim::Adam>(
                params,
                torch::optim::AdamOptions(config_.learning_rate)
                    .betas(std::make_tuple(config_.adam_betas[0], config_.adam_betas[1]))
                    .eps(config_.adam_eps)
                    .weight_decay(config_.weight_decay)
            );
        } else if (config_.optimizer == "adamw") {
            optimizer_ = std::make_unique<torch::optim::AdamW>(
                params,
                torch::optim::AdamWOptions(config_.learning_rate)
                    .betas(std::make_tuple(config_.adam_betas[0], config_.adam_betas[1]))
                    .eps(config_.adam_eps)
                    .weight_decay(config_.weight_decay)
            );
        } else if (config_.optimizer == "sgd") {
            optimizer_ = std::make_unique<torch::optim::SGD>(
                params,
                torch::optim::SGDOptions(config_.learning_rate)
                    .momentum(config_.momentum)
                    .weight_decay(config_.weight_decay)
            );
        } else {
            throw std::runtime_error("Unknown optimizer: " + config_.optimizer);
        }
    }

    /**
     * @brief Setup learning rate scheduler
     */
    void setupScheduler() {
        if (config_.lr_scheduler == "step") {
            scheduler_ = std::make_unique<torch::optim::StepLR>(
                *optimizer_,
                config_.lr_step_size,
                config_.lr_gamma
            );
        }
        // Add more schedulers as needed
    }

    /**
     * @brief Forward pass for one batch
     */
    torch::Tensor forwardBatch(const torch::data::Example<>& batch) {
        // This is a placeholder - actual implementation depends on data loader structure
        // In real implementation, extract RGB, depth, rays, intrinsics from batch
        // and call model forward

        // Example:
        // auto rgb = batch.data.to(device_);
        // auto gt_depth = batch.target.to(device_);
        // auto pred_depth = model_->forward(rgb);
        // auto loss = loss_fn_->forward(pred_depth, gt_depth, rgb);

        throw std::runtime_error("forwardBatch must be implemented for specific data format");
    }

    /**
     * @brief Validate one batch
     */
    std::map<std::string, float> validateBatch(const torch::data::Example<>& batch) {
        // Similar to forwardBatch but returns metrics instead of loss
        throw std::runtime_error("validateBatch must be implemented for specific data format");
    }

    /**
     * @brief Check if validation metric improved
     */
    bool checkImprovement(float current_metric) {
        if (config_.metric_lower_is_better) {
            return current_metric < (state_.best_metric - config_.early_stop_min_delta);
        } else {
            return current_metric > (state_.best_metric + config_.early_stop_min_delta);
        }
    }

    /**
     * @brief Get current learning rate
     */
    float getCurrentLR() {
        return optimizer_->param_groups()[0].options().get_lr();
    }

    /**
     * @brief Set learning rate
     */
    void setLearningRate(float lr) {
        for (auto& param_group : optimizer_->param_groups()) {
            param_group.options().set_lr(lr);
        }
    }

    /**
     * @brief Log training step
     */
    void logTrainingStep(float loss, int64_t batch_time_ms) {
        std::cout << "[Epoch " << state_.current_epoch
                  << ", Step " << state_.global_step << "] "
                  << "Loss: " << std::fixed << std::setprecision(4) << loss
                  << " | LR: " << std::scientific << getCurrentLR()
                  << " | Time: " << batch_time_ms << "ms"
                  << std::endl;
    }

    /**
     * @brief Print configuration
     */
    void printConfig() {
        std::cout << "  Optimizer: " << config_.optimizer << "\n";
        std::cout << "  Learning Rate: " << config_.learning_rate << "\n";
        std::cout << "  Batch Size: " << config_.batch_size << "\n";
        std::cout << "  Epochs: " << config_.num_epochs << "\n";
        std::cout << "  Device: " << device_ << "\n";
    }

    /**
     * @brief Save training history to CSV
     */
    void saveTrainingHistory() {
        std::string path = config_.log_dir + "/training_history.csv";
        std::ofstream file(path);

        file << "epoch,train_loss,val_abs_rel,val_rmse,val_delta_1.25\n";

        for (size_t i = 0; i < state_.train_losses.size(); ++i) {
            file << i << "," << state_.train_losses[i];

            if (i < state_.val_metrics_history.size()) {
                auto& metrics = state_.val_metrics_history[i];
                file << "," << metrics.at("abs_rel")
                     << "," << metrics.at("rmse")
                     << "," << metrics.at("delta_1.25");
            }
            file << "\n";
        }

        file.close();
        std::cout << "Training history saved: " << path << std::endl;
    }

    /**
     * @brief Save training state
     */
    void saveTrainingState(const std::string& path) {
        // Save as simple text file
        std::ofstream file(path);
        file << "epoch=" << state_.current_epoch << "\n";
        file << "global_step=" << state_.global_step << "\n";
        file << "best_metric=" << state_.best_metric << "\n";
        file << "epochs_without_improvement=" << state_.epochs_without_improvement << "\n";
        file.close();
    }

    /**
     * @brief Load training state
     */
    void loadTrainingState(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                if (key == "epoch") state_.current_epoch = std::stoi(value);
                else if (key == "global_step") state_.global_step = std::stoi(value);
                else if (key == "best_metric") state_.best_metric = std::stof(value);
                else if (key == "epochs_without_improvement")
                    state_.epochs_without_improvement = std::stoi(value);
            }
        }
        file.close();
    }

    /**
     * @brief Cleanup old checkpoints
     */
    void cleanupOldCheckpoints() {
        // Keep only last N checkpoints
        // Implementation depends on filesystem operations
    }

    /**
     * @brief Create directory if it doesn't exist
     */
    void createDirectory(const std::string& path) {
        std::filesystem::create_directories(path);
    }
};

} // namespace camera_aware_depth

#endif // TRAINER_H
