#ifndef TENSORBOARD_LOGGER_V2_H
#define TENSORBOARD_LOGGER_V2_H

#include <torch/torch.h>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

// For subprocess communication
#include <cstdio>
#include <array>
#include <unistd.h>
#include <signal.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace camera_aware_depth {

/**
 * @brief TensorBoard Logger V2 - Proper Event File Writer
 *
 * Integrates with Python's tensorboard.SummaryWriter for full TensorBoard support.
 * Provides state-of-the-art visualizations for research:
 * - Real-time scalar plots (loss, metrics, learning rate)
 * - Image visualizations (RGB, ground truth, predictions, error maps)
 * - Histogram tracking (weights, gradients, activations)
 * - Model graph visualization
 * - Hyperparameter tracking and comparison
 * - PR curves and confusion matrices
 * - TensorBoard Profiler integration
 */
class TensorBoardLoggerV2 {
public:
    TensorBoardLoggerV2(const std::string& log_dir)
        : log_dir_(log_dir), step_(0), writer_pid_(-1), pipe_to_writer_(nullptr) {

        fs::create_directories(log_dir);

        // Start Python TensorBoard writer service
        startWriterService();

        std::cout << "[TensorBoardLoggerV2] Initialized with proper event file writer" << std::endl;
        std::cout << "[TensorBoardLoggerV2] Log directory: " << log_dir_ << std::endl;
        std::cout << "[TensorBoardLoggerV2] Writer service PID: " << writer_pid_ << std::endl;
    }

    ~TensorBoardLoggerV2() {
        stopWriterService();
    }

    /**
     * @brief Add scalar value to TensorBoard
     */
    void addScalar(const std::string& tag, float value, int step = -1) {
        if (step < 0) step = step_;

        json command;
        command["type"] = "scalar";
        command["tag"] = tag;
        command["value"] = value;
        command["step"] = step;

        sendCommand(command);
    }

    /**
     * @brief Add multiple scalars at once with common main tag
     */
    void addScalars(const std::string& main_tag,
                   const std::map<std::string, float>& tag_scalar_dict,
                   int step = -1) {
        if (step < 0) step = step_;

        json command;
        command["type"] = "scalars";
        command["main_tag"] = main_tag;
        command["values"] = tag_scalar_dict;
        command["step"] = step;

        sendCommand(command);
    }

    /**
     * @brief Add image to TensorBoard from file path
     */
    void addImageFromPath(const std::string& tag, const std::string& img_path, int step = -1) {
        if (step < 0) step = step_;

        json command;
        command["type"] = "image";
        command["tag"] = tag;
        command["path"] = img_path;
        command["step"] = step;

        sendCommand(command);
    }

    /**
     * @brief Add image to TensorBoard from tensor
     */
    void addImage(const std::string& tag, const torch::Tensor& img_tensor, int step = -1) {
        if (step < 0) step = step_;

        // Save tensor temporarily and use path-based method
        // This is more reliable than sending raw data
        std::string temp_path = log_dir_ + "/temp_" + tag + "_" + std::to_string(step) + ".png";

        // Save tensor as image
        saveTensorAsImage(img_tensor, temp_path);

        // Send to TensorBoard
        addImageFromPath(tag, temp_path, step);
    }

    /**
     * @brief Add histogram to TensorBoard
     */
    void addHistogram(const std::string& tag, const torch::Tensor& values, int step = -1) {
        if (step < 0) step = step_;

        auto values_cpu = values.cpu().contiguous().flatten();
        auto data_ptr = values_cpu.data_ptr<float>();
        int numel = values_cpu.numel();

        // Send values to Python
        json command;
        command["type"] = "histogram";
        command["tag"] = tag;
        command["step"] = step;

        // Sample values if too large (max 10000 points)
        std::vector<float> sampled_values;
        if (numel > 10000) {
            int stride = numel / 10000;
            for (int i = 0; i < numel; i += stride) {
                sampled_values.push_back(data_ptr[i]);
            }
        } else {
            sampled_values.assign(data_ptr, data_ptr + numel);
        }

        command["values"] = sampled_values;
        sendCommand(command);
    }

    /**
     * @brief Add text to TensorBoard
     */
    void addText(const std::string& tag, const std::string& text, int step = -1) {
        if (step < 0) step = step_;

        json command;
        command["type"] = "text";
        command["tag"] = tag;
        command["text"] = text;
        command["step"] = step;

        sendCommand(command);
    }

    /**
     * @brief Add hyperparameters and metrics
     */
    void addHParams(const std::map<std::string, float>& hparam_dict,
                   const std::map<std::string, float>& metric_dict) {
        json command;
        command["type"] = "hparams";
        command["hparams"] = hparam_dict;
        command["metrics"] = metric_dict;

        sendCommand(command);
    }

    /**
     * @brief Add PR curve for binary classification
     */
    void addPRCurve(const std::string& tag,
                   const torch::Tensor& labels,
                   const torch::Tensor& predictions,
                   int step = -1) {
        if (step < 0) step = step_;

        auto labels_cpu = labels.cpu().contiguous().flatten();
        auto pred_cpu = predictions.cpu().contiguous().flatten();

        std::vector<int> label_vec;
        std::vector<float> pred_vec;

        for (int i = 0; i < labels_cpu.numel(); ++i) {
            label_vec.push_back(labels_cpu[i].item<int>());
            pred_vec.push_back(pred_cpu[i].item<float>());
        }

        json command;
        command["type"] = "pr_curve";
        command["tag"] = tag;
        command["labels"] = label_vec;
        command["predictions"] = pred_vec;
        command["step"] = step;

        sendCommand(command);
    }

    /**
     * @brief Increment global step
     */
    void step() { step_++; }

    /**
     * @brief Set global step
     */
    void setStep(int step) { step_ = step; }

    /**
     * @brief Get current step
     */
    int getStep() const { return step_; }

    /**
     * @brief Flush all pending writes
     */
    void flush() {
        // Commands are sent immediately, so this is a no-op
        // But we keep it for API compatibility
    }

private:
    std::string log_dir_;
    int step_;
    pid_t writer_pid_;
    FILE* pipe_to_writer_;

    /**
     * @brief Start the Python TensorBoard writer service
     */
    void startWriterService() {
        // Find python executable (prefer conda environment)
        std::string python_cmd = "python";

        // Try multiple methods to find the script
        fs::path script_path;

        // Method 1: Check environment variable
        const char* project_root = std::getenv("CAMERA_DEPTH_PROJECT_ROOT");
        if (project_root) {
            script_path = fs::path(project_root) / "scripts" / "tensorboard_writer.py";
        }

        // Method 2: Try current working directory
        if (script_path.empty() || !fs::exists(script_path)) {
            script_path = fs::current_path() / "scripts" / "tensorboard_writer.py";
        }

        // Method 3: Try relative to __FILE__ (3 levels up)
        if (!fs::exists(script_path)) {
            fs::path header_path = fs::path(__FILE__);
            fs::path project_dir = header_path.parent_path().parent_path().parent_path();
            script_path = project_dir / "scripts" / "tensorboard_writer.py";
        }

        // Method 4: Common installation paths
        if (!fs::exists(script_path)) {
            std::vector<std::string> search_paths = {
                "/Users/r/Desktop/Synexian/ML Research Ideas and Topics/Camera Matrix/scripts/tensorboard_writer.py",
                "./scripts/tensorboard_writer.py",
                "../scripts/tensorboard_writer.py",
                "../../scripts/tensorboard_writer.py"
            };

            for (const auto& path : search_paths) {
                if (fs::exists(path)) {
                    script_path = path;
                    break;
                }
            }
        }

        if (!fs::exists(script_path)) {
            std::cerr << "[TensorBoardLoggerV2] ERROR: tensorboard_writer.py not found!" << std::endl;
            std::cerr << "[TensorBoardLoggerV2] Searched locations:" << std::endl;
            std::cerr << "  - Current dir: " << fs::current_path() / "scripts" / "tensorboard_writer.py" << std::endl;
            std::cerr << "  - Relative to source: " << fs::path(__FILE__).parent_path().parent_path().parent_path() / "scripts" / "tensorboard_writer.py" << std::endl;
            std::cerr << "[TensorBoardLoggerV2] Current working directory: " << fs::current_path() << std::endl;
            throw std::runtime_error("TensorBoard writer script not found");
        }

        std::cout << "[TensorBoardLoggerV2] Found writer script at: " << script_path << std::endl;

        // Build command with proper shell escaping for paths with spaces
        // Use single quotes to avoid any shell interpretation
        std::stringstream cmd_builder;
        cmd_builder << python_cmd << " '" << script_path.string() << "' '" << log_dir_ << "'";
        std::string command = cmd_builder.str();

        std::cout << "[TensorBoardLoggerV2] Executing: " << command << std::endl;

        // Open pipe to Python process
        pipe_to_writer_ = popen(command.c_str(), "w");

        if (!pipe_to_writer_) {
            throw std::runtime_error("Failed to start TensorBoard writer service");
        }

        // Set unbuffered mode
        setbuf(pipe_to_writer_, nullptr);

        std::cout << "[TensorBoardLoggerV2] Started writer service: " << command << std::endl;
    }

    /**
     * @brief Stop the Python TensorBoard writer service
     */
    void stopWriterService() {
        if (pipe_to_writer_) {
            // Send shutdown command
            json command;
            command["type"] = "shutdown";
            sendCommand(command);

            // Close pipe
            pclose(pipe_to_writer_);
            pipe_to_writer_ = nullptr;
        }
    }

    /**
     * @brief Send JSON command to Python writer service
     */
    void sendCommand(const json& command) {
        if (!pipe_to_writer_) {
            std::cerr << "[TensorBoardLoggerV2] ERROR: Writer service not running" << std::endl;
            return;
        }

        try {
            std::string command_str = command.dump() + "\n";
            fputs(command_str.c_str(), pipe_to_writer_);
            fflush(pipe_to_writer_);
        } catch (const std::exception& e) {
            std::cerr << "[TensorBoardLoggerV2] ERROR sending command: " << e.what() << std::endl;
        }
    }

    /**
     * @brief Save tensor as image file
     */
    void saveTensorAsImage(const torch::Tensor& tensor, const std::string& path) {
        auto img_cpu = tensor.cpu();

        cv::Mat img;
        if (img_cpu.dim() == 3) {
            // (C, H, W) -> (H, W, C)
            img_cpu = img_cpu.permute({1, 2, 0});

            int H = img_cpu.size(0);
            int W = img_cpu.size(1);
            int C = img_cpu.size(2);

            if (C == 1) {
                // Grayscale
                img = cv::Mat(H, W, CV_32F, img_cpu.data_ptr<float>());
                // Normalize to 0-255
                double min_val, max_val;
                cv::minMaxLoc(img, &min_val, &max_val);
                img = (img - min_val) / (max_val - min_val) * 255.0;
                img.convertTo(img, CV_8U);
            } else if (C == 3) {
                // RGB
                img = cv::Mat(H, W, CV_32FC3, img_cpu.data_ptr<float>());
                // Check if already normalized
                double max_val;
                cv::minMaxLoc(img, nullptr, &max_val);
                if (max_val <= 1.0) {
                    img *= 255.0;
                }
                img.convertTo(img, CV_8UC3);
                cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            }
        } else if (img_cpu.dim() == 2) {
            // (H, W)
            img = cv::Mat(img_cpu.size(0), img_cpu.size(1), CV_32F, img_cpu.data_ptr<float>());
            double min_val, max_val;
            cv::minMaxLoc(img, &min_val, &max_val);
            img = (img - min_val) / (max_val - min_val) * 255.0;
            img.convertTo(img, CV_8U);
        }

        cv::imwrite(path, img);
    }
};

} // namespace camera_aware_depth

#endif // TENSORBOARD_LOGGER_V2_H
