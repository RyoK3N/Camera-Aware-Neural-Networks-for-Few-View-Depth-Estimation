#ifndef PRODUCTION_TRAINER_H
#define PRODUCTION_TRAINER_H

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <filesystem>
#include "../data/sunrgbd_loader.h"
#include "../loss/depth_loss.h"
#include "../evaluation/depth_metrics.h"

namespace fs = std::filesystem;

namespace camera_aware_depth {

/**
 * @brief Production-ready trainer with batching and comprehensive logging
 */
class ProductionTrainer {
public:
    struct Config {
        // Training
        int num_epochs = 50;
        int batch_size = 8;
        float learning_rate = 1e-4f;
        float weight_decay = 1e-5f;
        bool use_grad_clip = true;
        float grad_clip_value = 1.0f;

        // Validation
        int val_interval = 1;
        int log_interval = 10;
        int save_interval = 5;

        // Paths
        std::string checkpoint_dir = "./checkpoints";
        std::string log_dir = "./logs";
        std::string experiment_name = "experiment";

        // Device
        torch::Device device = torch::kCPU;
    };

    struct TrainingMetrics {
        float loss = 0.0f;
        float si_loss = 0.0f;
        float grad_loss = 0.0f;
        float smooth_loss = 0.0f;
        int samples_processed = 0;
        float learning_rate = 0.0f;
    };

    ProductionTrainer(
        std::shared_ptr<torch::nn::Module> model,
        std::shared_ptr<BaselineUNetImpl> model_impl,
        std::shared_ptr<CombinedDepthLoss> loss_fn,
        const Config& config
    ) : model_(model),
        model_impl_(model_impl),
        loss_fn_(loss_fn),
        config_(config) {

        // Create directories
        fs::create_directories(config_.checkpoint_dir);
        fs::create_directories(config_.log_dir);

        // Initialize optimizer
        optimizer_ = std::make_shared<torch::optim::Adam>(
            model_->parameters(),
            torch::optim::AdamOptions(config_.learning_rate)
                .weight_decay(config_.weight_decay)
        );

        // Open log files
        std::string train_log_path = config_.log_dir + "/training.log";
        std::string metrics_csv_path = config_.log_dir + "/metrics.csv";

        train_log_.open(train_log_path, std::ios::app);
        metrics_csv_.open(metrics_csv_path, std::ios::app);

        // Write CSV header if file is new
        if (metrics_csv_.tellp() == 0) {
            metrics_csv_ << "epoch,step,train_loss,train_si_loss,train_grad_loss,train_smooth_loss,"
                        << "val_loss,val_abs_rel,val_rmse,learning_rate,time_elapsed\n";
        }

        logMessage("Trainer initialized");
        logMessage("Checkpoint dir: " + config_.checkpoint_dir);
        logMessage("Log dir: " + config_.log_dir);
    }

    ~ProductionTrainer() {
        if (train_log_.is_open()) train_log_.close();
        if (metrics_csv_.is_open()) metrics_csv_.close();
    }

    /**
     * @brief Train with proper batching and validation
     */
    void train(
        std::shared_ptr<SunRGBDLoader> train_loader,
        std::shared_ptr<SunRGBDLoader> val_loader
    ) {
        logMessage("\n=== Starting Training ===");
        logMessage("Train samples: " + std::to_string(train_loader->size()));
        logMessage("Val samples: " + std::to_string(val_loader->size()));
        logMessage("Batch size: " + std::to_string(config_.batch_size));
        logMessage("Epochs: " + std::to_string(config_.num_epochs));

        auto start_time = std::chrono::high_resolution_clock::now();
        int global_step = 0;

        for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
            logMessage("\n--- Epoch " + std::to_string(epoch + 1) + "/" +
                      std::to_string(config_.num_epochs) + " ---");

            // Training
            auto train_metrics = trainEpoch(train_loader, global_step);
            global_step += train_metrics.samples_processed / config_.batch_size;

            // Validation
            TrainingMetrics val_metrics;
            if ((epoch + 1) % config_.val_interval == 0) {
                val_metrics = validateEpoch(val_loader);
            }

            // Log metrics
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();

            logEpochMetrics(epoch + 1, global_step, train_metrics, val_metrics, elapsed);

            // Save checkpoint
            if ((epoch + 1) % config_.save_interval == 0) {
                saveCheckpoint(epoch + 1, train_metrics.loss);
            }
        }

        // Save final model
        saveCheckpoint(config_.num_epochs, 0.0f, "final_model.pt");
        logMessage("\n=== Training Complete ===");
    }

private:
    std::shared_ptr<torch::nn::Module> model_;
    std::shared_ptr<BaselineUNetImpl> model_impl_;
    std::shared_ptr<CombinedDepthLoss> loss_fn_;
    std::shared_ptr<torch::optim::Adam> optimizer_;
    Config config_;

    std::ofstream train_log_;
    std::ofstream metrics_csv_;

    /**
     * @brief Train for one epoch with batching
     */
    TrainingMetrics trainEpoch(std::shared_ptr<SunRGBDLoader> loader, int global_step) {
        model_->train();

        TrainingMetrics metrics;
        int num_samples = loader->size();
        int num_batches = (num_samples + config_.batch_size - 1) / config_.batch_size;

        std::cout << "Training: ";
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Create batch
            int start_idx = batch_idx * config_.batch_size;
            int end_idx = std::min(start_idx + config_.batch_size, num_samples);
            int actual_batch_size = end_idx - start_idx;

            std::vector<size_t> indices;
            for (int i = start_idx; i < end_idx; ++i) {
                indices.push_back(i);
            }

            auto samples = loader->getBatch(indices);

            // Prepare batch tensors
            std::vector<torch::Tensor> rgb_list, depth_list, intrinsics_list;
            for (const auto& sample : samples) {
                rgb_list.push_back(sample.rgb);
                depth_list.push_back(sample.depth);
                intrinsics_list.push_back(sample.intrinsics);
            }

            auto rgb_batch = torch::stack(rgb_list).to(config_.device);
            auto depth_batch = torch::stack(depth_list).to(config_.device);
            auto intrinsics_batch = torch::stack(intrinsics_list).to(config_.device);

            // Forward pass
            optimizer_->zero_grad();

            // Model forward using concrete implementation
            auto depth_pred = model_impl_->forward(rgb_batch);

            // Compute loss (with reprojection error using camera intrinsics)
            auto loss = loss_fn_->forwardWithIntrinsics(depth_pred, depth_batch, rgb_batch, intrinsics_batch);

            // Backward pass
            loss.backward();

            if (config_.use_grad_clip) {
                torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.grad_clip_value);
            }

            optimizer_->step();

            // Update metrics
            float loss_val = loss.item<float>();
            metrics.loss += loss_val * actual_batch_size;
            metrics.samples_processed += actual_batch_size;

            // Progress
            if ((batch_idx + 1) % config_.log_interval == 0 || batch_idx == num_batches - 1) {
                float progress = 100.0f * (batch_idx + 1) / num_batches;
                std::cout << "\r  [" << std::setw(3) << (int)progress << "%] "
                          << "Batch " << (batch_idx + 1) << "/" << num_batches
                          << " | Loss: " << std::fixed << std::setprecision(4) << loss_val
                          << std::flush;
            }
        }

        std::cout << std::endl;

        // Average metrics
        if (metrics.samples_processed > 0) {
            metrics.loss /= metrics.samples_processed;
        }

        return metrics;
    }

    /**
     * @brief Validate for one epoch
     */
    TrainingMetrics validateEpoch(std::shared_ptr<SunRGBDLoader> loader) {
        model_->eval();

        TrainingMetrics metrics;
        int num_samples = std::min(500, (int)loader->size());  // Validate on subset

        std::cout << "Validation: ";
        torch::NoGradGuard no_grad;

        for (int i = 0; i < num_samples; ++i) {
            auto sample = loader->getSample(i);

            auto rgb = sample.rgb.unsqueeze(0).to(config_.device);
            auto depth_gt = sample.depth.unsqueeze(0).to(config_.device);
            auto intrinsics = sample.intrinsics.unsqueeze(0).to(config_.device);

            // Forward pass using concrete implementation
            auto depth_pred = model_impl_->forward(rgb);

            auto loss = loss_fn_->forwardWithIntrinsics(depth_pred, depth_gt, rgb, intrinsics);

            metrics.loss += loss.item<float>();
            metrics.samples_processed++;

            if ((i + 1) % 50 == 0 || i == num_samples - 1) {
                float progress = 100.0f * (i + 1) / num_samples;
                std::cout << "\r  [" << std::setw(3) << (int)progress << "%] "
                          << "Sample " << (i + 1) << "/" << num_samples
                          << std::flush;
            }
        }

        std::cout << std::endl;

        if (metrics.samples_processed > 0) {
            metrics.loss /= metrics.samples_processed;
        }

        return metrics;
    }

    /**
     * @brief Log epoch metrics to console and files
     */
    void logEpochMetrics(
        int epoch,
        int step,
        const TrainingMetrics& train_metrics,
        const TrainingMetrics& val_metrics,
        long elapsed_seconds
    ) {
        std::stringstream ss;
        ss << "Epoch " << epoch
           << " | Train Loss: " << std::fixed << std::setprecision(4) << train_metrics.loss;

        if (val_metrics.samples_processed > 0) {
            ss << " | Val Loss: " << std::fixed << std::setprecision(4) << val_metrics.loss;
        }

        ss << " | Time: " << elapsed_seconds << "s";

        logMessage(ss.str());

        // Write to CSV
        metrics_csv_ << epoch << ","
                    << step << ","
                    << train_metrics.loss << ","
                    << train_metrics.si_loss << ","
                    << train_metrics.grad_loss << ","
                    << train_metrics.smooth_loss << ","
                    << val_metrics.loss << ","
                    << 0.0f << ","  // abs_rel placeholder
                    << 0.0f << ","  // rmse placeholder
                    << config_.learning_rate << ","
                    << elapsed_seconds << "\n";
        metrics_csv_.flush();
    }

    /**
     * @brief Save checkpoint
     */
    void saveCheckpoint(int epoch, float loss, const std::string& name = "") {
        std::string checkpoint_name = name.empty() ?
            "checkpoint_epoch_" + std::to_string(epoch) + ".pt" : name;
        std::string checkpoint_path = config_.checkpoint_dir + "/" + checkpoint_name;

        torch::save(model_, checkpoint_path);
        logMessage("Saved checkpoint: " + checkpoint_path);
    }

    /**
     * @brief Log message to console and file
     */
    void logMessage(const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = std::localtime(&time_t);

        std::stringstream timestamp;
        timestamp << std::put_time(tm, "%Y-%m-%d %H:%M:%S");

        std::string log_line = "[" + timestamp.str() + "] " + message;

        std::cout << log_line << std::endl;
        if (train_log_.is_open()) {
            train_log_ << log_line << "\n";
            train_log_.flush();
        }
    }
};

} // namespace camera_aware_depth

#endif // PRODUCTION_TRAINER_H
