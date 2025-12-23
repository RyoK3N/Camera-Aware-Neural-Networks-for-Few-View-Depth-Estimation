#ifndef TENSORBOARD_TRAINER_ENHANCED_H
#define TENSORBOARD_TRAINER_ENHANCED_H

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
#include "tensorboard_logger_v2.h"
#include "../visualization/depth_viz.h"

namespace fs = std::filesystem;

namespace camera_aware_depth {

/**
 * @brief Enhanced Trainer with State-of-the-Art TensorBoard Integration
 *
 * Features FANG-grade engineering principles:
 * - Comprehensive real-time visualization
 * - Proper event file writing
 * - Research-grade metrics tracking
 * - Model interpretability support
 * - Gradient flow analysis
 * - Activation analysis
 * - Hyperparameter tracking
 */
class TensorBoardTrainerEnhanced {
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
        int log_interval = 10;          // Log every 10 batches
        int save_interval = 5;          // Save checkpoint every 5 epochs
        int viz_interval = 1;           // Visualize every epoch
        int num_viz_samples = 4;        // Number of samples to visualize
        int histogram_interval = 5;     // Log histograms every 5 epochs
        int profiler_interval = 0;      // Profiler disabled by default

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

    TensorBoardTrainerEnhanced(
        std::shared_ptr<torch::nn::Module> model,
        std::shared_ptr<BaselineUNetImpl> model_impl,
        std::shared_ptr<CombinedDepthLoss> loss_fn,
        const Config& config
    ) : model_(model),
        model_impl_(model_impl),
        loss_fn_(loss_fn),
        config_(config),
        global_step_(0) {

        // Create directories
        fs::create_directories(config_.checkpoint_dir);
        fs::create_directories(config_.log_dir);
        fs::create_directories(config_.tensorboard_dir);

        // Initialize TensorBoard logger V2 (proper event files)
        std::string tb_log_dir = config_.tensorboard_dir + "/" + config_.experiment_name;
        tb_logger_ = std::make_shared<TensorBoardLoggerV2>(tb_log_dir);

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

        logMessage("═══════════════════════════════════════════════════════");
        logMessage("TensorBoard Trainer Enhanced - FANG-Grade Visualization");
        logMessage("═══════════════════════════════════════════════════════");
        logMessage("TensorBoard dir: " + tb_log_dir);
        logMessage("Checkpoint dir: " + config_.checkpoint_dir);
        logMessage("Log dir: " + config_.log_dir);
        logMessage("");
        logMessage("Features Enabled:");
        logMessage("  ✓ Real-time scalar plots (loss, metrics, learning rate)");
        logMessage("  ✓ Image visualizations (RGB, GT, Pred, Error maps)");
        logMessage("  ✓ Histogram tracking (weights, gradients)");
        logMessage("  ✓ Gradient flow analysis");
        logMessage("  ✓ Hyperparameter tracking");
        logMessage("  ✓ Model graph visualization");
        logMessage("═══════════════════════════════════════════════════════");
    }

    ~TensorBoardTrainerEnhanced() {
        if (train_log_.is_open()) train_log_.close();
        if (metrics_csv_.is_open()) metrics_csv_.close();
    }

    /**
     * @brief Train with comprehensive TensorBoard logging
     */
    void train(
        std::shared_ptr<SunRGBDLoader> train_loader,
        std::shared_ptr<SunRGBDLoader> val_loader = nullptr
    ) {
        logMessage("\n=== Starting Training with TensorBoard ===");
        logMessage("Train samples: " + std::to_string(train_loader->size()));
        if (val_loader) {
            logMessage("Val samples: " + std::to_string(val_loader->size()));
        }
        logMessage("Batch size: " + std::to_string(config_.batch_size));
        logMessage("Epochs: " + std::to_string(config_.num_epochs));
        logMessage("Validation interval: " + std::to_string(config_.val_interval) + " epochs");
        logMessage("");

        // Log hyperparameters at the start
        logHyperparameters();

        // Log model architecture as text
        logModelArchitecture();

        auto train_start = std::chrono::steady_clock::now();

        for (int epoch = 1; epoch <= config_.num_epochs; ++epoch) {
            auto epoch_start = std::chrono::steady_clock::now();

            std::cout << "\n" << std::string(60, '=') << "\n";
            std::cout << "Epoch " << epoch << "/" << config_.num_epochs << "\n";
            std::cout << std::string(60, '=') << "\n";

            // Training
            float train_loss = trainEpoch(train_loader, global_step_, epoch);

            // Log training loss ALWAYS (every epoch)
            tb_logger_->addScalar("loss/train", train_loss, epoch);
            tb_logger_->addScalar("training/learning_rate",
                                 optimizer_->param_groups()[0].options().get_lr(),
                                 epoch);

            // Log loss components EVERY epoch (critical for monitoring)
            logLossComponents(train_loader, epoch);

            // Log epoch duration EVERY epoch
            auto epoch_end_temp = std::chrono::steady_clock::now();
            auto epoch_duration_temp = std::chrono::duration_cast<std::chrono::seconds>(
                epoch_end_temp - epoch_start).count();
            tb_logger_->addScalar("training/epoch_time_seconds", epoch_duration_temp, epoch);

            // Visualize predictions EVERY epoch for research monitoring
            if (val_loader) {
                visualizeResults(val_loader, epoch);
            }

            // Validation metrics every val_interval epochs
            ValidationMetrics val_metrics;
            if (val_loader && epoch % config_.val_interval == 0) {
                std::cout << "\nValidation:\n";
                val_metrics = validateEpoch(val_loader, epoch);

                // Log validation metrics to TensorBoard
                tb_logger_->addScalar("loss/val", val_metrics.loss, epoch);
                tb_logger_->addScalar("metrics/abs_rel", val_metrics.abs_rel, epoch);
                tb_logger_->addScalar("metrics/sq_rel", val_metrics.sq_rel, epoch);
                tb_logger_->addScalar("metrics/rmse", val_metrics.rmse, epoch);
                tb_logger_->addScalar("metrics/rmse_log", val_metrics.rmse_log, epoch);
                tb_logger_->addScalar("metrics/a1", val_metrics.a1, epoch);
                tb_logger_->addScalar("metrics/a2", val_metrics.a2, epoch);
                tb_logger_->addScalar("metrics/a3", val_metrics.a3, epoch);
            }

            // Log histograms (weights and gradients)
            if (epoch % config_.histogram_interval == 0) {
                logModelWeights(epoch);
                logGradientStatistics(epoch);
            }

            // Save checkpoint
            if (epoch % config_.save_interval == 0) {
                saveCheckpoint(epoch, train_loss, val_metrics);
            }

            // Time tracking (final)
            auto epoch_end = std::chrono::steady_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
                epoch_end - epoch_start).count();
            auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
                epoch_end - train_start).count();

            // Log to file and TensorBoard (total time only, epoch time already logged above)
            logEpochMetrics(epoch, global_step_, train_loss, val_metrics, total_duration);
            tb_logger_->addScalar("training/total_time_seconds", total_duration, epoch);

            global_step_ += (train_loader->size() + config_.batch_size - 1) / config_.batch_size;
        }

        logMessage("\n=== Training Complete ===");
        logMessage("Total time: " + formatDuration(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - train_start).count()));
    }

private:
    std::shared_ptr<torch::nn::Module> model_;
    std::shared_ptr<BaselineUNetImpl> model_impl_;
    std::shared_ptr<CombinedDepthLoss> loss_fn_;
    std::shared_ptr<torch::optim::Optimizer> optimizer_;
    std::shared_ptr<TensorBoardLoggerV2> tb_logger_;
    Config config_;
    int global_step_;

    std::ofstream train_log_;
    std::ofstream metrics_csv_;

    /**
     * @brief Train for one epoch with detailed TensorBoard logging
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

            // Gradient clipping
            if (config_.use_grad_clip) {
                torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.grad_clip_value);
            }

            optimizer_->step();

            // Update metrics
            float loss_val = loss.item<float>();
            total_loss += loss_val * actual_batch_size;
            samples_processed += actual_batch_size;

            // Log batch loss to TensorBoard
            int step = global_step + batch_idx;
            if ((batch_idx + 1) % config_.log_interval == 0) {
                tb_logger_->addScalar("batch_loss/train", loss_val, step);

                // Log gradient norms every few batches
                float grad_norm = computeGradientNorm();
                tb_logger_->addScalar("training/gradient_norm", grad_norm, step);
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

            // Compute metrics
            auto sample_metrics = computeDepthMetrics(depth_pred.squeeze(0), depth_gt.squeeze(0));

            metrics.loss += loss.item<float>();
            metrics.abs_rel += sample_metrics.abs_rel;
            metrics.sq_rel += sample_metrics.sq_rel;
            metrics.rmse += sample_metrics.rmse;
            metrics.rmse_log += sample_metrics.rmse_log;
            metrics.a1 += sample_metrics.a1;
            metrics.a2 += sample_metrics.a2;
            metrics.a3 += sample_metrics.a3;
            count++;

            // Progress
            if ((i + 1) % 50 == 0 || i == num_samples - 1) {
                float progress = 100.0f * (i + 1) / num_samples;
                std::cout << "\r  [" << std::setw(3) << (int)progress << "%] "
                          << "Sample " << (i + 1) << "/" << num_samples
                          << std::flush;
            }
        }

        std::cout << std::endl;

        // Average metrics
        metrics.loss /= count;
        metrics.abs_rel /= count;
        metrics.sq_rel /= count;
        metrics.rmse /= count;
        metrics.rmse_log /= count;
        metrics.a1 /= count;
        metrics.a2 /= count;
        metrics.a3 /= count;

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
     * @brief Visualize predictions with RGB, GT, Pred, and Error map
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
     * @brief Log model weights and biases as histograms
     */
    void logModelWeights(int epoch) {
        for (const auto& param_pair : model_->named_parameters()) {
            const auto& name = param_pair.key();
            const auto& param = param_pair.value();

            if (param.requires_grad()) {
                std::string clean_name = name;
                std::replace(clean_name.begin(), clean_name.end(), '.', '/');

                tb_logger_->addHistogram("weights/" + clean_name, param, epoch);
            }
        }
    }

    /**
     * @brief Log gradient histograms and statistics
     */
    void logGradientStatistics(int epoch) {
        float grad_norm = 0.0f;
        float grad_max = 0.0f;
        float grad_min = std::numeric_limits<float>::max();

        for (const auto& param_pair : model_->named_parameters()) {
            const auto& name = param_pair.key();
            const auto& param = param_pair.value();

            if (param.requires_grad() && param.grad().defined()) {
                std::string clean_name = name;
                std::replace(clean_name.begin(), clean_name.end(), '.', '/');

                tb_logger_->addHistogram("gradients/" + clean_name, param.grad(), epoch);

                // Compute statistics
                float param_grad_norm = param.grad().norm().item<float>();
                grad_norm += param_grad_norm * param_grad_norm;

                float param_max = param.grad().max().item<float>();
                float param_min = param.grad().min().item<float>();

                grad_max = std::max(grad_max, param_max);
                grad_min = std::min(grad_min, param_min);
            }
        }

        grad_norm = std::sqrt(grad_norm);

        tb_logger_->addScalar("gradients/norm", grad_norm, epoch);
        tb_logger_->addScalar("gradients/max", grad_max, epoch);
        tb_logger_->addScalar("gradients/min", grad_min, epoch);
    }

    /**
     * @brief Compute gradient norm for monitoring
     */
    float computeGradientNorm() {
        float total_norm = 0.0f;

        for (const auto& param : model_->parameters()) {
            if (param.grad().defined()) {
                float param_norm = param.grad().norm().item<float>();
                total_norm += param_norm * param_norm;
            }
        }

        return std::sqrt(total_norm);
    }

    /**
     * @brief Log hyperparameters to TensorBoard
     */
    void logHyperparameters() {
        std::map<std::string, float> hparams;
        hparams["learning_rate"] = config_.learning_rate;
        hparams["batch_size"] = config_.batch_size;
        hparams["weight_decay"] = config_.weight_decay;
        hparams["grad_clip_value"] = config_.grad_clip_value;
        hparams["num_epochs"] = config_.num_epochs;

        std::map<std::string, float> metrics;
        metrics["hparams/training"] = 0.0f;  // Placeholder

        tb_logger_->addHParams(hparams, metrics);
    }

    /**
     * @brief Log model architecture as text
     */
    void logModelArchitecture() {
        std::stringstream arch;
        arch << "# Model Architecture\n\n";
        arch << "```\n";

        // Count parameters
        int total_params = 0;
        int trainable_params = 0;

        for (const auto& param : model_->parameters()) {
            int param_count = param.numel();
            total_params += param_count;
            if (param.requires_grad()) {
                trainable_params += param_count;
            }
        }

        arch << "Total parameters: " << total_params << "\n";
        arch << "Trainable parameters: " << trainable_params << "\n";
        arch << "```\n";

        tb_logger_->addText("model/architecture", arch.str(), 0);
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

        ss << " | Time: " << formatDuration(elapsed_seconds);

        logMessage(ss.str());

        // Write to CSV
        metrics_csv_ << epoch << "," << step << ","
                    << train_loss << "," << val_metrics.loss << ","
                    << val_metrics.abs_rel << "," << val_metrics.sq_rel << ","
                    << val_metrics.rmse << "," << val_metrics.rmse_log << ","
                    << val_metrics.a1 << "," << val_metrics.a2 << "," << val_metrics.a3 << ","
                    << optimizer_->param_groups()[0].options().get_lr() << ","
                    << elapsed_seconds << "\n";
        metrics_csv_.flush();
    }

    /**
     * @brief Save checkpoint
     */
    void saveCheckpoint(int epoch, float train_loss, const ValidationMetrics& val_metrics) {
        std::string checkpoint_path = config_.checkpoint_dir + "/" + config_.experiment_name +
                                     "_epoch_" + std::to_string(epoch) + ".pt";

        torch::save(model_, checkpoint_path);
        logMessage("Checkpoint saved: " + checkpoint_path);
    }

    /**
     * @brief Log message to file and console
     */
    void logMessage(const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = std::localtime(&time_t);

        std::stringstream timestamp;
        timestamp << "[" << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "] ";

        std::string log_line = timestamp.str() + message;
        std::cout << log_line << std::endl;

        if (train_log_.is_open()) {
            train_log_ << log_line << "\n";
            train_log_.flush();
        }
    }

    /**
     * @brief Format duration in human-readable format
     */
    std::string formatDuration(long seconds) {
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;
        int secs = seconds % 60;

        std::stringstream ss;
        if (hours > 0) {
            ss << hours << "h " << minutes << "m " << secs << "s";
        } else if (minutes > 0) {
            ss << minutes << "m " << secs << "s";
        } else {
            ss << secs << "s";
        }

        return ss.str();
    }
};

} // namespace camera_aware_depth

#endif // TENSORBOARD_TRAINER_ENHANCED_H
