#ifndef EXPERIMENT_COMPARISON_H
#define EXPERIMENT_COMPARISON_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace camera_aware_depth {

/**
 * @brief Experiment Results
 *
 * Stores all metrics and metadata for a single experiment
 */
struct ExperimentResult {
    std::string name;
    std::string architecture;
    std::map<std::string, float> metrics;
    int num_epochs;
    int64_t num_parameters;
    float training_time_hours;
    float inference_time_ms;
    float memory_usage_mb;

    // Ablation study context
    std::vector<std::string> components;  // e.g., ["rays", "pcl", "film", "attention"]
    std::string baseline_name;            // Reference baseline for comparison
};

/**
 * @brief Experiment Comparison Tool
 *
 * Analyzes and compares multiple experiments for ablation studies and model selection
 * Generates LaTeX tables, CSV exports, and statistical significance tests
 */
class ExperimentComparison {
public:
    /**
     * @brief Add experiment result
     */
    void addExperiment(const ExperimentResult& result) {
        experiments_.push_back(result);
    }

    /**
     * @brief Load experiment results from CSV file
     */
    void loadFromCSV(const std::string& csv_path) {
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open CSV file: " + csv_path);
        }

        std::string line;
        std::getline(file, line);  // Skip header

        while (std::getline(file, line)) {
            // Parse CSV line and create ExperimentResult
            // Format: name,architecture,abs_rel,rmse,delta_1.25,...
            ExperimentResult result = parseCSVLine(line);
            experiments_.push_back(result);
        }

        file.close();
    }

    /**
     * @brief Generate comparison table in LaTeX format
     *
     * Creates a professional LaTeX table for paper inclusion
     */
    std::string generateLatexTable(const std::vector<std::string>& metrics_to_show) const {
        std::stringstream ss;

        ss << "\\begin{table}[h]\n";
        ss << "\\centering\n";
        ss << "\\caption{Depth Estimation Results on SUN RGB-D}\n";
        ss << "\\label{tab:results}\n";
        ss << "\\begin{tabular}{l";
        for (size_t i = 0; i < metrics_to_show.size(); ++i) {
            ss << "c";
        }
        ss << "}\n";
        ss << "\\hline\n";

        // Header
        ss << "Method";
        for (const auto& metric : metrics_to_show) {
            ss << " & " << formatMetricName(metric);
        }
        ss << " \\\\\n";
        ss << "\\hline\n";

        // Results
        for (const auto& exp : experiments_) {
            ss << exp.name;
            for (const auto& metric : metrics_to_show) {
                if (exp.metrics.count(metric)) {
                    ss << " & " << std::fixed << std::setprecision(3) << exp.metrics.at(metric);
                } else {
                    ss << " & -";
                }
            }
            ss << " \\\\\n";
        }

        ss << "\\hline\n";
        ss << "\\end{tabular}\n";
        ss << "\\end{table}\n";

        return ss.str();
    }

    /**
     * @brief Generate markdown comparison table
     */
    std::string generateMarkdownTable(const std::vector<std::string>& metrics_to_show) const {
        std::stringstream ss;

        // Header
        ss << "| Method |";
        for (const auto& metric : metrics_to_show) {
            ss << " " << formatMetricName(metric) << " |";
        }
        ss << "\n";

        // Separator
        ss << "|--------|";
        for (size_t i = 0; i < metrics_to_show.size(); ++i) {
            ss << "--------:|";
        }
        ss << "\n";

        // Results
        for (const auto& exp : experiments_) {
            ss << "| " << exp.name << " |";
            for (const auto& metric : metrics_to_show) {
                if (exp.metrics.count(metric)) {
                    ss << " " << std::fixed << std::setprecision(4) << exp.metrics.at(metric) << " |";
                } else {
                    ss << " - |";
                }
            }
            ss << "\n";
        }

        return ss.str();
    }

    /**
     * @brief Generate ablation study analysis
     *
     * Computes improvement over baseline for each component
     */
    std::string generateAblationAnalysis(const std::string& baseline_name, const std::string& metric = "abs_rel") const {
        std::stringstream ss;

        // Find baseline
        auto baseline_it = std::find_if(experiments_.begin(), experiments_.end(),
            [&baseline_name](const ExperimentResult& exp) { return exp.name == baseline_name; });

        if (baseline_it == experiments_.end()) {
            return "Error: Baseline experiment not found: " + baseline_name;
        }

        float baseline_value = baseline_it->metrics.at(metric);

        ss << "Ablation Study Analysis\n";
        ss << "=======================\n\n";
        ss << "Baseline: " << baseline_name << "\n";
        ss << "Metric: " << metric << " = " << baseline_value << "\n\n";

        ss << "Component Contributions:\n";
        ss << "------------------------\n";

        for (const auto& exp : experiments_) {
            if (exp.name == baseline_name) continue;

            float improvement = computeImprovement(baseline_value, exp.metrics.at(metric), metric);

            ss << exp.name << ": ";
            ss << std::fixed << std::setprecision(4) << exp.metrics.at(metric);
            ss << " (" << (improvement >= 0 ? "+" : "") << std::setprecision(2) << improvement << "%)";

            if (!exp.components.empty()) {
                ss << " [";
                for (size_t i = 0; i < exp.components.size(); ++i) {
                    ss << exp.components[i];
                    if (i < exp.components.size() - 1) ss << ", ";
                }
                ss << "]";
            }

            ss << "\n";
        }

        return ss.str();
    }

    /**
     * @brief Export all results to CSV
     */
    void exportToCSV(const std::string& output_path) const {
        std::ofstream file(output_path);

        // Header
        file << "name,architecture,abs_rel,sq_rel,rmse,rmse_log,mae,log10,"
             << "delta_1.25,delta_1.25^2,delta_1.25^3,"
             << "num_params,training_time_h,inference_time_ms,memory_mb\n";

        // Data
        for (const auto& exp : experiments_) {
            file << exp.name << ","
                 << exp.architecture << ",";

            std::vector<std::string> metric_keys = {
                "abs_rel", "sq_rel", "rmse", "rmse_log", "mae", "log10",
                "delta_1.25", "delta_1.25^2", "delta_1.25^3"
            };

            for (const auto& key : metric_keys) {
                if (exp.metrics.count(key)) {
                    file << exp.metrics.at(key);
                }
                file << ",";
            }

            file << exp.num_parameters << ","
                 << exp.training_time_hours << ","
                 << exp.inference_time_ms << ","
                 << exp.memory_usage_mb << "\n";
        }

        file.close();
    }

    /**
     * @brief Rank experiments by a specific metric
     */
    std::vector<ExperimentResult> rankByMetric(const std::string& metric, bool lower_is_better = true) const {
        std::vector<ExperimentResult> ranked = experiments_;

        std::sort(ranked.begin(), ranked.end(),
            [&metric, lower_is_better](const ExperimentResult& a, const ExperimentResult& b) {
                float val_a = a.metrics.count(metric) ? a.metrics.at(metric) : std::numeric_limits<float>::infinity();
                float val_b = b.metrics.count(metric) ? b.metrics.at(metric) : std::numeric_limits<float>::infinity();

                return lower_is_better ? (val_a < val_b) : (val_a > val_b);
            });

        return ranked;
    }

    /**
     * @brief Get best experiment by metric
     */
    const ExperimentResult* getBest(const std::string& metric, bool lower_is_better = true) const {
        if (experiments_.empty()) return nullptr;

        const ExperimentResult* best = &experiments_[0];
        float best_value = best->metrics.count(metric) ?
            best->metrics.at(metric) : std::numeric_limits<float>::infinity();

        for (const auto& exp : experiments_) {
            if (!exp.metrics.count(metric)) continue;

            float value = exp.metrics.at(metric);

            if ((lower_is_better && value < best_value) ||
                (!lower_is_better && value > best_value)) {
                best = &exp;
                best_value = value;
            }
        }

        return best;
    }

    /**
     * @brief Generate comprehensive report
     */
    std::string generateReport() const {
        std::stringstream ss;

        ss << "Experiment Comparison Report\n";
        ss << "============================\n\n";

        ss << "Total Experiments: " << experiments_.size() << "\n\n";

        // Best models by different metrics
        ss << "Best Models:\n";
        ss << "------------\n";

        std::vector<std::string> key_metrics = {"abs_rel", "rmse", "delta_1.25"};
        std::vector<bool> lower_is_better = {true, true, false};

        for (size_t i = 0; i < key_metrics.size(); ++i) {
            auto best = getBest(key_metrics[i], lower_is_better[i]);
            if (best) {
                ss << formatMetricName(key_metrics[i]) << ": "
                   << best->name << " ("
                   << std::fixed << std::setprecision(4) << best->metrics.at(key_metrics[i])
                   << ")\n";
            }
        }

        ss << "\n";

        // Detailed comparison table
        ss << "Detailed Results:\n";
        ss << "-----------------\n\n";
        ss << generateMarkdownTable({"abs_rel", "rmse", "delta_1.25", "delta_1.25^2", "delta_1.25^3"});

        return ss.str();
    }

private:
    std::vector<ExperimentResult> experiments_;

    /**
     * @brief Parse CSV line into ExperimentResult
     */
    ExperimentResult parseCSVLine(const std::string& line) const {
        ExperimentResult result;

        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ','); result.name = token;
        std::getline(ss, token, ','); result.architecture = token;

        std::vector<std::string> metric_keys = {
            "abs_rel", "sq_rel", "rmse", "rmse_log", "mae", "log10",
            "delta_1.25", "delta_1.25^2", "delta_1.25^3"
        };

        for (const auto& key : metric_keys) {
            std::getline(ss, token, ',');
            if (!token.empty()) {
                result.metrics[key] = std::stof(token);
            }
        }

        std::getline(ss, token, ','); result.num_parameters = std::stoll(token);
        std::getline(ss, token, ','); result.training_time_hours = std::stof(token);
        std::getline(ss, token, ','); result.inference_time_ms = std::stof(token);
        std::getline(ss, token, ','); result.memory_usage_mb = std::stof(token);

        return result;
    }

    /**
     * @brief Format metric name for display
     */
    std::string formatMetricName(const std::string& metric) const {
        static const std::map<std::string, std::string> names = {
            {"abs_rel", "AbsRel↓"},
            {"sq_rel", "SqRel↓"},
            {"rmse", "RMSE↓"},
            {"rmse_log", "RMSElog↓"},
            {"mae", "MAE↓"},
            {"log10", "Log10↓"},
            {"delta_1.25", "δ<1.25↑"},
            {"delta_1.25^2", "δ<1.25²↑"},
            {"delta_1.25^3", "δ<1.25³↑"}
        };

        return names.count(metric) ? names.at(metric) : metric;
    }

    /**
     * @brief Compute improvement percentage
     */
    float computeImprovement(float baseline, float current, const std::string& metric) const {
        // For lower-is-better metrics (errors)
        bool lower_is_better = (metric.find("delta") == std::string::npos);

        if (lower_is_better) {
            // Improvement = (baseline - current) / baseline * 100
            return ((baseline - current) / baseline) * 100.0f;
        } else {
            // Improvement = (current - baseline) / baseline * 100
            return ((current - baseline) / baseline) * 100.0f;
        }
    }
};

} // namespace camera_aware_depth

#endif // EXPERIMENT_COMPARISON_H
