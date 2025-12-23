#ifndef TENSORBOARD_LOGGER_H
#define TENSORBOARD_LOGGER_H

#include <torch/torch.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace camera_aware_depth {

/**
 * @brief Simple TensorBoard logger for C++
 *
 * Writes events in TensorBoard's event file format for visualization
 * Note: This is a simplified version. For full TensorBoard support,
 * consider using tensorboardX or similar libraries.
 */
class TensorBoardLogger {
public:
    TensorBoardLogger(const std::string& log_dir)
        : log_dir_(log_dir), step_(0) {

        fs::create_directories(log_dir);

        // Create log file with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = std::localtime(&time_t);

        std::stringstream filename;
        filename << log_dir_ << "/events_"
                << std::put_time(tm, "%Y%m%d_%H%M%S") << ".txt";

        log_file_.open(filename.str(), std::ios::app);

        // Also create simple CSV for easy plotting
        scalar_log_.open(log_dir_ + "/scalars.csv", std::ios::app);
        scalar_log_ << "step,tag,value\n";
    }

    ~TensorBoardLogger() {
        if (log_file_.is_open()) log_file_.close();
        if (scalar_log_.is_open()) scalar_log_.close();
    }

    /**
     * @brief Log a scalar value
     */
    void addScalar(const std::string& tag, float value, int step = -1) {
        if (step < 0) step = step_;

        auto timestamp = getCurrentTimestamp();

        // Write to text log
        log_file_ << "[" << timestamp << "] "
                 << "SCALAR step=" << step
                 << " tag=" << tag
                 << " value=" << std::fixed << std::setprecision(6) << value
                 << "\n";
        log_file_.flush();

        // Write to CSV for easy plotting
        scalar_log_ << step << "," << tag << "," << value << "\n";
        scalar_log_.flush();
    }

    /**
     * @brief Log multiple scalars at once
     */
    void addScalars(const std::string& main_tag,
                   const std::map<std::string, float>& tag_scalar_dict,
                   int step = -1) {
        if (step < 0) step = step_;

        for (const auto& [tag, value] : tag_scalar_dict) {
            addScalar(main_tag + "/" + tag, value, step);
        }
    }

    /**
     * @brief Log an image (saves as PNG)
     */
    void addImage(const std::string& tag,
                 const torch::Tensor& img_tensor,
                 int step = -1) {
        if (step < 0) step = step_;

        // Create images directory
        std::string img_dir = log_dir_ + "/images";
        fs::create_directories(img_dir);

        // Save image
        std::stringstream filename;
        filename << img_dir << "/" << tag << "_step_" << step << ".png";

        // Convert tensor to cv::Mat and save
        auto img_cpu = img_tensor.cpu();

        // Assume tensor is (C, H, W) or (H, W)
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
                img.convertTo(img, CV_8U, 255.0);
            } else if (C == 3) {
                // RGB
                img = cv::Mat(H, W, CV_32FC3, img_cpu.data_ptr<float>());
                img.convertTo(img, CV_8UC3, 255.0);
                cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            }
        } else {
            // (H, W)
            img = cv::Mat(img_cpu.size(0), img_cpu.size(1), CV_32F,
                         img_cpu.data_ptr<float>());
            img.convertTo(img, CV_8U, 255.0);
        }

        cv::imwrite(filename.str(), img);

        // Log to file
        auto timestamp = getCurrentTimestamp();
        log_file_ << "[" << timestamp << "] "
                 << "IMAGE step=" << step
                 << " tag=" << tag
                 << " file=" << filename.str() << "\n";
        log_file_.flush();
    }

    /**
     * @brief Log a histogram (simplified - just saves statistics)
     */
    void addHistogram(const std::string& tag,
                     const torch::Tensor& values,
                     int step = -1) {
        if (step < 0) step = step_;

        auto values_cpu = values.cpu().contiguous();
        auto data = values_cpu.data_ptr<float>();
        int numel = values_cpu.numel();

        // Compute statistics
        float mean = values_cpu.mean().item<float>();
        float std = values_cpu.std().item<float>();
        float min = values_cpu.min().item<float>();
        float max = values_cpu.max().item<float>();

        auto timestamp = getCurrentTimestamp();
        log_file_ << "[" << timestamp << "] "
                 << "HISTOGRAM step=" << step
                 << " tag=" << tag
                 << " mean=" << mean
                 << " std=" << std
                 << " min=" << min
                 << " max=" << max
                 << " numel=" << numel << "\n";
        log_file_.flush();
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

private:
    std::string log_dir_;
    int step_;
    std::ofstream log_file_;
    std::ofstream scalar_log_;

    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = std::localtime(&time_t);

        std::stringstream ss;
        ss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

} // namespace camera_aware_depth

#endif // TENSORBOARD_LOGGER_H
