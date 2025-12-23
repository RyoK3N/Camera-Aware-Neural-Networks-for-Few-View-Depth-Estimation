#ifndef TENSORBOARD_TRAINER_H
#define TENSORBOARD_TRAINER_H

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
#include "tensorboard_logger.h"
#include "../visualization/depth_viz.h"

namespace fs = std::filesystem;

namespace camera_aware_depth {

/**
 * @brief Enhanced trainer with TensorBoard logging and visualization
 */
class TensorBoardTrainer {
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
        int val_interval = 10;          // Validate every 10 epochs
        int log_interval = 10;
        int save_interval = 5;
        int viz_interval = 1;           // Visualize every epoch
        int num_viz_samples = 4;        // Number of samples to visualize

        // Paths
        std::string checkpoint_dir = "./checkpoints";
        std::string log_dir = "./logs";
        std::string tensorboard_dir = "./runs";
        std::string experiment_name = "experiment";

        // Device
        torch::Device device = torch::kCPU;
    };

    struct ValidationMetrics {
        float loss = 0.0f;
        float abs_rel = 0.0f;
        float sq_rel = 0.0f;
        float rmse = 0.0f;
        float rmse_log = 0.0f;
        float a1 = 0.0f;  // delta < 1.25
        float a2 = 0.0f;  // delta < 1.25^2
        float a3 = 0.0f;  // delta < 1.25^3
    };

    TensorBoardTrainer(
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
        fs::create_directories(config_.tensorboard_dir);

        // Initialize TensorBoard logger
        std::string tb_log_dir = config_.tensorboard_dir + "/" + config_.experiment_name;
        tb_logger_ = std::make_shared<TensorBoardLogger>(tb_log_dir);

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
            metrics_csv_ << "epoch,step,train_loss,val_loss,"
                        << "abs_rel,sq_rel,rmse,rmse_log,a1,a2,a3,"
                        << "learning_rate,time_elapsed\n";
        }

        logMessage("TensorBoard Trainer initialized");
        logMessage("TensorBoard dir: " + tb_log_dir);
        logMessage("Checkpoint dir: " + config_.checkpoint_dir);
        logMessage("Log dir: " + config_.log_dir);
    }

    ~TensorBoardTrainer() {
        if (train_log_.is_open()) train_log_.close();
        if (metrics_csv_.is_open()) metrics_csv_.close();
    }

    /**
     * @brief Train with TensorBoard logging and visualization
     */
    void train(
        std::shared_ptr<SunRGBDLoader> train_loader,
        std::shared_ptr<SunRGBDLoader> val_loader
    ) {
        logMessage("\n=== Starting Training with TensorBoard ===");
        logMessage("Train samples: " + std::to_string(train_loader->size()));
        logMessage("Val samples: " + std::to_string(val_loader->size()));
        logMessage("Batch size: " + std::to_string(config_.batch_size));
        logMessage("Epochs: " + std::to_string(config_.num_epochs));
        logMessage("Validation interval: " + std::to_string(config_.val_interval) + " epochs");

        auto start_time = std::chrono::high_resolution_clock::now();
        int global_step = 0;

        for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
            logMessage("\n--- Epoch " + std::to_string(epoch + 1) + "/" +
                      std::to_string(config_.num_epochs) + " ---");

            // Training
            float train_loss = trainEpoch(train_loader, global_step, epoch + 1);
            global_step += train_loader->size() / config_.batch_size;

            // Log training loss
            tb_logger_->addScalar("loss/train", train_loss, epoch + 1);
            tb_logger_->addScalar("learning_rate", config_.learning_rate, epoch + 1);

            // Log individual loss components (sample from first batch)
            logLossComponents(train_loader, epoch + 1);

            // Validation
            ValidationMetrics val_metrics;
            if ((epoch + 1) % config_.val_interval == 0) {
                logMessage("Running validation...");
                val_metrics = validateEpoch(val_loader, epoch + 1);

                // Log validation metrics to TensorBoard
                tb_logger_->addScalar("loss/val", val_metrics.loss, epoch + 1);
                tb_logger_->addScalar("metrics/abs_rel", val_metrics.abs_rel, epoch + 1);
                tb_logger_->addScalar("metrics/sq_rel", val_metrics.sq_rel, epoch + 1);
                tb_logger_->addScalar("metrics/rmse", val_metrics.rmse, epoch + 1);
                tb_logger_->addScalar("metrics/rmse_log", val_metrics.rmse_log, epoch + 1);
                tb_logger_->addScalar("metrics/a1", val_metrics.a1, epoch + 1);
                tb_logger_->addScalar("metrics/a2", val_metrics.a2, epoch + 1);
                tb_logger_->addScalar("metrics/a3", val_metrics.a3, epoch + 1);
            }

            // Visualization
            if ((epoch + 1) % config_.viz_interval == 0) {
                visualizeResults(val_loader, epoch + 1);
            }

            // Log epoch metrics
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - start_time).count();

            logEpochMetrics(epoch + 1, global_step, train_loss, val_metrics, elapsed);

            // Save checkpoint
            if ((epoch + 1) % config_.save_interval == 0) {
                saveCheckpoint(epoch + 1, train_loss);
            }

            // Log model weights histogram
            if ((epoch + 1) % config_.val_interval == 0) {
                logModelWeights(epoch + 1);
            }
        }

        // Save final model
        saveCheckpoint(config_.num_epochs, 0.0f, "final_model.pt");
        logMessage("\n=== Training Complete ===");
        logMessage("View TensorBoard logs:");
        logMessage("  tensorboard --logdir=" + config_.tensorboard_dir);
    }

private:
    std::shared_ptr<torch::nn::Module> model_;
    std::shared_ptr<BaselineUNetImpl> model_impl_;
    std::shared_ptr<CombinedDepthLoss> loss_fn_;
    std::shared_ptr<torch::optim::Adam> optimizer_;
    std::shared_ptr<TensorBoardLogger> tb_logger_;
    Config config_;

    std::ofstream train_log_;
    std::ofstream metrics_csv_;

    /**
     * @brief Train for one epoch with TensorBoard logging
     */
    float trainEpoch(std::shared_ptr<SunRGBDLoader> loader, int global_step, int epoch) {
        model_->train();

        float total_loss = 0.0f;
        int num_samples = loader->size();
        int num_batches = (num_samples + config_.batch_size - 1) / config_.batch_size;
        int samples_processed = 0;

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
            auto depth_pred = model_impl_->forward(rgb_batch);
            auto loss = loss_fn_->forwardWithIntrinsics(depth_pred, depth_batch, rgb_batch, intrinsics_batch);

            // Backward pass
            loss.backward();

            if (config_.use_grad_clip) {
                torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.grad_clip_value);
            }

            optimizer_->step();

            // Update metrics
            float loss_val = loss.item<float>();
            total_loss += loss_val * actual_batch_size;
            samples_processed += actual_batch_size;

            // Log to TensorBoard
            int step = global_step + batch_idx;
            if ((batch_idx + 1) % config_.log_interval == 0) {
                tb_logger_->addScalar("batch_loss/train", loss_val, step);
            }

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

        return total_loss / samples_processed;
    }

    /**
     * @brief Validate with comprehensive metrics
     */
    ValidationMetrics validateEpoch(std::shared_ptr<SunRGBDLoader> loader, int epoch) {
        model_->eval();

        ValidationMetrics metrics;
        int num_samples = std::min(500, (int)loader->size());
        int count = 0;

        std::cout << "Validation: ";
        torch::NoGradGuard no_grad;

        for (int i = 0; i < num_samples; ++i) {
            auto sample = loader->getSample(i);

            auto rgb = sample.rgb.unsqueeze(0).to(config_.device);
            auto depth_gt = sample.depth.unsqueeze(0).to(config_.device);
            auto intrinsics = sample.intrinsics.unsqueeze(0).to(config_.device);

            // Forward pass
            auto depth_pred = model_impl_->forward(rgb);
            auto loss = loss_fn_->forwardWithIntrinsics(depth_pred, depth_gt, rgb, intrinsics);

            // Compute depth metrics
            auto depth_metrics = computeDepthMetrics(depth_pred, depth_gt);

            metrics.loss += loss.item<float>();
            metrics.abs_rel += depth_metrics.abs_rel;
            metrics.sq_rel += depth_metrics.sq_rel;
            metrics.rmse += depth_metrics.rmse;
            metrics.rmse_log += depth_metrics.rmse_log;
            metrics.a1 += depth_metrics.a1;
            metrics.a2 += depth_metrics.a2;
            metrics.a3 += depth_metrics.a3;
            count++;

            if ((i + 1) % 50 == 0 || i == num_samples - 1) {
                float progress = 100.0f * (i + 1) / num_samples;
                std::cout << "\r  [" << std::setw(3) << (int)progress << "%] "
                          << "Sample " << (i + 1) << "/" << num_samples
                          << std::flush;
            }
        }

        std::cout << std::endl;

        // Average metrics
        if (count > 0) {
            metrics.loss /= count;
            metrics.abs_rel /= count;
            metrics.sq_rel /= count;
            metrics.rmse /= count;
            metrics.rmse_log /= count;
            metrics.a1 /= count;
            metrics.a2 /= count;
            metrics.a3 /= count;
        }

        return metrics;
    }

    /**
     * @brief Compute depth estimation metrics
     */
    ValidationMetrics computeDepthMetrics(
        const torch::Tensor& pred,
        const torch::Tensor& gt
    ) {
        ValidationMetrics metrics;

        auto pred_flat = pred.view({-1});
        auto gt_flat = gt.view({-1});

        // Valid mask (depth > 0)
        auto valid_mask = gt_flat > 0.0f;
        auto pred_valid = pred_flat.masked_select(valid_mask);
        auto gt_valid = gt_flat.masked_select(valid_mask);

        if (pred_valid.numel() == 0) {
            return metrics;
        }

        // Absolute relative error
        auto abs_diff = torch::abs(pred_valid - gt_valid);
        metrics.abs_rel = (abs_diff / gt_valid).mean().item<float>();

        // Squared relative error
        metrics.sq_rel = ((abs_diff * abs_diff) / gt_valid).mean().item<float>();

        // RMSE
        metrics.rmse = torch::sqrt((abs_diff * abs_diff).mean()).item<float>();

        // RMSE log
        auto log_diff = torch::abs(torch::log(pred_valid + 1e-8) - torch::log(gt_valid + 1e-8));
        metrics.rmse_log = torch::sqrt((log_diff * log_diff).mean()).item<float>();

        // Threshold accuracy
        auto ratio = torch::max(pred_valid / gt_valid, gt_valid / pred_valid);
        metrics.a1 = (ratio < 1.25f).to(torch::kFloat32).mean().item<float>();
        metrics.a2 = (ratio < 1.5625f).to(torch::kFloat32).mean().item<float>();  // 1.25^2
        metrics.a3 = (ratio < 1.953125f).to(torch::kFloat32).mean().item<float>(); // 1.25^3

        return metrics;
    }

    /**
     * @brief Visualize prediction results
     */
    void visualizeResults(std::shared_ptr<SunRGBDLoader> loader, int epoch) {
        model_->eval();
        torch::NoGradGuard no_grad;

        for (int i = 0; i < config_.num_viz_samples && i < (int)loader->size(); ++i) {
            auto sample = loader->getSample(i);

            auto rgb = sample.rgb.unsqueeze(0).to(config_.device);
            auto depth_gt = sample.depth.unsqueeze(0).to(config_.device);

            // Forward pass
            auto depth_pred = model_impl_->forward(rgb);

            // Create visualization
            auto viz = DepthVisualizer::createComparisonViz(
                depth_pred.squeeze(0).cpu(),
                depth_gt.squeeze(0).cpu(),
                sample.rgb
            );

            // Log to TensorBoard
            std::string tag = "predictions/sample_" + std::to_string(i);
            tb_logger_->addImage(tag, viz, epoch);
        }

        logMessage("Visualizations saved for epoch " + std::to_string(epoch));
    }

    /**
     * @brief Log individual loss components to TensorBoard
     */
    void logLossComponents(std::shared_ptr<SunRGBDLoader> loader, int epoch) {
        model_->eval();
        torch::NoGradGuard no_grad;

        // Sample from first batch
        auto sample = loader->getSample(0);

        auto rgb = sample.rgb.unsqueeze(0).to(config_.device);
        auto depth_gt = sample.depth.unsqueeze(0).to(config_.device);
        auto intrinsics = sample.intrinsics.unsqueeze(0).to(config_.device);

        // Forward pass
        auto depth_pred = model_impl_->forward(rgb);

        // Get loss components
        auto components = loss_fn_->getComponentsWithIntrinsics(
            depth_pred, depth_gt, rgb, intrinsics
        );

        // Log each component
        tb_logger_->addScalar("loss_components/si_loss", components["si_loss"], epoch);
        tb_logger_->addScalar("loss_components/grad_loss", components["grad_loss"], epoch);
        tb_logger_->addScalar("loss_components/smooth_loss", components["smooth_loss"], epoch);
        tb_logger_->addScalar("loss_components/reproj_loss", components["reproj_loss"], epoch);

        model_->train();
    }

    /**
     * @brief Log model weights as histograms
     */
    void logModelWeights(int epoch) {
        for (const auto& param_pair : model_->named_parameters()) {
            const auto& name = param_pair.key();
            const auto& param = param_pair.value();

            if (param.requires_grad()) {
                tb_logger_->addHistogram("weights/" + name, param, epoch);

                if (param.grad().defined()) {
                    tb_logger_->addHistogram("gradients/" + name, param.grad(), epoch);
                }
            }
        }
    }

    /**
     * @brief Log epoch metrics
     */
    void logEpochMetrics(
        int epoch,
        int step,
        float train_loss,
        const ValidationMetrics& val_metrics,
        long elapsed_seconds
    ) {
        std::stringstream ss;
        ss << "Epoch " << epoch
           << " | Train Loss: " << std::fixed << std::setprecision(4) << train_loss;

        if (val_metrics.loss > 0) {
            ss << " | Val Loss: " << val_metrics.loss
               << " | abs_rel: " << val_metrics.abs_rel
               << " | rmse: " << val_metrics.rmse
               << " | δ<1.25: " << std::setprecision(3) << val_metrics.a1;
        }

        ss << " | Time: " << elapsed_seconds << "s";

        logMessage(ss.str());

        // Write to CSV
        metrics_csv_ << epoch << ","
                    << step << ","
                    << train_loss << ","
                    << val_metrics.loss << ","
                    << val_metrics.abs_rel << ","
                    << val_metrics.sq_rel << ","
                    << val_metrics.rmse << ","
                    << val_metrics.rmse_log << ","
                    << val_metrics.a1 << ","
                    << val_metrics.a2 << ","
                    << val_metrics.a3 << ","
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
     * @brief Log message
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

#endif // TENSORBOARD_TRAINER_H
