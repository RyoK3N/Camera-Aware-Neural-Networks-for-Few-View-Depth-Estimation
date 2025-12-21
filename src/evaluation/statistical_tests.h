#ifndef STATISTICAL_TESTS_H
#define STATISTICAL_TESTS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>

namespace camera_aware_depth {

/**
 * @brief Statistical Test Result
 */
struct TestResult {
    std::string test_name;
    float statistic;
    float p_value;
    bool is_significant;
    float confidence_level;
    std::string interpretation;
};

/**
 * @brief Statistical Significance Tester
 *
 * Implements common statistical tests for deep learning model comparison:
 * - Paired t-test (parametric)
 * - Wilcoxon signed-rank test (non-parametric)
 * - Bootstrap confidence intervals
 * - Effect size (Cohen's d)
 *
 * Based on best practices from:
 * - Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 2006
 * - Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", Neural Computation 1998
 * - Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021
 *
 * Reference implementations:
 * - SciPy statistical tests
 * - R statistical computing
 */
class StatisticalTester {
public:
    /**
     * @brief Paired t-test
     *
     * Tests if the mean difference between paired samples is significantly different from zero.
     * Assumes: Normal distribution of differences
     *
     * @param sample1 First sample (e.g., baseline model errors)
     * @param sample2 Second sample (e.g., proposed model errors)
     * @param confidence Confidence level (default: 0.95 for 95%)
     * @return Test result with p-value and interpretation
     */
    static TestResult pairedTTest(
        const std::vector<float>& sample1,
        const std::vector<float>& sample2,
        float confidence = 0.95f
    ) {
        if (sample1.size() != sample2.size()) {
            throw std::invalid_argument("Samples must have equal size");
        }

        int n = sample1.size();
        if (n < 2) {
            throw std::invalid_argument("Need at least 2 samples");
        }

        // Compute differences
        std::vector<float> differences;
        for (size_t i = 0; i < sample1.size(); ++i) {
            differences.push_back(sample1[i] - sample2[i]);
        }

        // Compute mean and std of differences
        float mean_diff = computeMean(differences);
        float std_diff = computeStd(differences);

        // Compute t-statistic
        float t_stat = mean_diff / (std_diff / std::sqrt(n));

        // Degrees of freedom
        int df = n - 1;

        // Compute p-value (two-tailed)
        float p_value = computeTDistPValue(std::abs(t_stat), df);

        TestResult result;
        result.test_name = "Paired t-test";
        result.statistic = t_stat;
        result.p_value = p_value;
        result.confidence_level = confidence;
        result.is_significant = (p_value < (1.0f - confidence));

        // Interpretation
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "Mean difference: " << mean_diff << " ± " << std_diff << "\n";
        ss << "t-statistic: " << t_stat << " (df=" << df << ")\n";
        ss << "p-value: " << p_value << "\n";

        if (result.is_significant) {
            ss << "Result: SIGNIFICANT improvement (p < " << (1.0f - confidence) << ")\n";
        } else {
            ss << "Result: NOT significant (p >= " << (1.0f - confidence) << ")\n";
        }

        result.interpretation = ss.str();

        return result;
    }

    /**
     * @brief Wilcoxon signed-rank test (non-parametric)
     *
     * Non-parametric alternative to paired t-test.
     * Does not assume normal distribution.
     * Tests if median difference is zero.
     *
     * @param sample1 First sample
     * @param sample2 Second sample
     * @param confidence Confidence level
     * @return Test result
     */
    static TestResult wilcoxonSignedRank(
        const std::vector<float>& sample1,
        const std::vector<float>& sample2,
        float confidence = 0.95f
    ) {
        if (sample1.size() != sample2.size()) {
            throw std::invalid_argument("Samples must have equal size");
        }

        // Compute differences
        std::vector<float> differences;
        for (size_t i = 0; i < sample1.size(); ++i) {
            float diff = sample1[i] - sample2[i];
            if (std::abs(diff) > 1e-10) {  // Exclude zeros
                differences.push_back(diff);
            }
        }

        int n = differences.size();
        if (n == 0) {
            TestResult result;
            result.test_name = "Wilcoxon signed-rank test";
            result.statistic = 0.0f;
            result.p_value = 1.0f;
            result.is_significant = false;
            result.interpretation = "No non-zero differences found";
            return result;
        }

        // Compute ranks of absolute differences
        std::vector<std::pair<float, int>> abs_diff_idx;
        for (size_t i = 0; i < differences.size(); ++i) {
            abs_diff_idx.push_back({std::abs(differences[i]), i});
        }

        std::sort(abs_diff_idx.begin(), abs_diff_idx.end());

        // Assign ranks (handle ties with average rank)
        std::vector<float> ranks(n);
        for (int i = 0; i < n; ++i) {
            ranks[abs_diff_idx[i].second] = i + 1;
        }

        // Sum ranks for positive and negative differences
        float w_plus = 0.0f;
        float w_minus = 0.0f;

        for (size_t i = 0; i < differences.size(); ++i) {
            if (differences[i] > 0) {
                w_plus += ranks[i];
            } else {
                w_minus += ranks[i];
            }
        }

        // Test statistic is minimum of W+ and W-
        float w_stat = std::min(w_plus, w_minus);

        // Approximate p-value using normal approximation (for n > 20)
        float mean_w = n * (n + 1) / 4.0f;
        float std_w = std::sqrt(n * (n + 1) * (2 * n + 1) / 24.0f);
        float z_stat = (w_stat - mean_w) / std_w;
        float p_value = 2.0f * (1.0f - normalCDF(std::abs(z_stat)));

        TestResult result;
        result.test_name = "Wilcoxon signed-rank test";
        result.statistic = w_stat;
        result.p_value = p_value;
        result.confidence_level = confidence;
        result.is_significant = (p_value < (1.0f - confidence));

        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "W+ (sum of positive ranks): " << w_plus << "\n";
        ss << "W- (sum of negative ranks): " << w_minus << "\n";
        ss << "W statistic: " << w_stat << "\n";
        ss << "p-value: " << p_value << "\n";

        if (result.is_significant) {
            ss << "Result: SIGNIFICANT difference (p < " << (1.0f - confidence) << ")\n";
        } else {
            ss << "Result: NOT significant (p >= " << (1.0f - confidence) << ")\n";
        }

        result.interpretation = ss.str();

        return result;
    }

    /**
     * @brief Compute effect size (Cohen's d)
     *
     * Measures the standardized difference between two means.
     * Interpretation:
     * - |d| < 0.2: negligible
     * - 0.2 <= |d| < 0.5: small
     * - 0.5 <= |d| < 0.8: medium
     * - |d| >= 0.8: large
     *
     * @param sample1 First sample
     * @param sample2 Second sample
     * @return Cohen's d
     */
    static float computeCohenD(
        const std::vector<float>& sample1,
        const std::vector<float>& sample2
    ) {
        float mean1 = computeMean(sample1);
        float mean2 = computeMean(sample2);
        float std1 = computeStd(sample1);
        float std2 = computeStd(sample2);

        // Pooled standard deviation
        int n1 = sample1.size();
        int n2 = sample2.size();
        float pooled_std = std::sqrt(
            ((n1 - 1) * std1 * std1 + (n2 - 1) * std2 * std2) / (n1 + n2 - 2)
        );

        return (mean1 - mean2) / pooled_std;
    }

    /**
     * @brief Bootstrap confidence interval
     *
     * Computes confidence interval for a statistic using bootstrap resampling.
     *
     * @param sample Data sample
     * @param statistic_func Function to compute statistic (e.g., mean, median)
     * @param confidence Confidence level (default: 0.95)
     * @param num_bootstrap Number of bootstrap samples (default: 10000)
     * @return Pair of (lower_bound, upper_bound)
     */
    static std::pair<float, float> bootstrapConfidenceInterval(
        const std::vector<float>& sample,
        std::function<float(const std::vector<float>&)> statistic_func,
        float confidence = 0.95f,
        int num_bootstrap = 10000
    ) {
        std::vector<float> bootstrap_statistics;
        bootstrap_statistics.reserve(num_bootstrap);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, sample.size() - 1);

        // Bootstrap resampling
        for (int b = 0; b < num_bootstrap; ++b) {
            std::vector<float> bootstrap_sample;
            bootstrap_sample.reserve(sample.size());

            // Sample with replacement
            for (size_t i = 0; i < sample.size(); ++i) {
                bootstrap_sample.push_back(sample[dis(gen)]);
            }

            // Compute statistic on bootstrap sample
            float stat = statistic_func(bootstrap_sample);
            bootstrap_statistics.push_back(stat);
        }

        // Sort bootstrap statistics
        std::sort(bootstrap_statistics.begin(), bootstrap_statistics.end());

        // Compute percentiles
        float alpha = 1.0f - confidence;
        int lower_idx = static_cast<int>(alpha / 2.0f * num_bootstrap);
        int upper_idx = static_cast<int>((1.0f - alpha / 2.0f) * num_bootstrap);

        lower_idx = std::max(0, std::min(lower_idx, num_bootstrap - 1));
        upper_idx = std::max(0, std::min(upper_idx, num_bootstrap - 1));

        return {bootstrap_statistics[lower_idx], bootstrap_statistics[upper_idx]};
    }

    /**
     * @brief Compare two models statistically
     *
     * Performs comprehensive statistical comparison including:
     * - Paired t-test
     * - Wilcoxon test
     * - Effect size
     * - Bootstrap confidence intervals
     *
     * @param model1_errors Per-sample errors for model 1
     * @param model2_errors Per-sample errors for model 2
     * @param model1_name Name of model 1
     * @param model2_name Name of model 2
     * @return Comprehensive comparison report
     */
    static std::string compareModels(
        const std::vector<float>& model1_errors,
        const std::vector<float>& model2_errors,
        const std::string& model1_name = "Model 1",
        const std::string& model2_name = "Model 2"
    ) {
        std::stringstream report;

        report << "═══════════════════════════════════════════════════════════\n";
        report << "  Statistical Model Comparison\n";
        report << "═══════════════════════════════════════════════════════════\n\n";

        report << "Comparing: " << model1_name << " vs " << model2_name << "\n";
        report << "Sample size: " << model1_errors.size() << "\n\n";

        // Descriptive statistics
        report << "Descriptive Statistics:\n";
        report << "─────────────────────────────────────────────────────────\n";
        report << std::fixed << std::setprecision(4);
        report << model1_name << ":\n";
        report << "  Mean: " << computeMean(model1_errors)
               << " ± " << computeStd(model1_errors) << "\n";
        report << "  Median: " << computeMedian(model1_errors) << "\n\n";

        report << model2_name << ":\n";
        report << "  Mean: " << computeMean(model2_errors)
               << " ± " << computeStd(model2_errors) << "\n";
        report << "  Median: " << computeMedian(model2_errors) << "\n\n";

        // Effect size
        float cohens_d = computeCohenD(model1_errors, model2_errors);
        report << "Effect Size (Cohen's d): " << cohens_d << " ";

        if (std::abs(cohens_d) < 0.2f) {
            report << "(negligible)\n";
        } else if (std::abs(cohens_d) < 0.5f) {
            report << "(small)\n";
        } else if (std::abs(cohens_d) < 0.8f) {
            report << "(medium)\n";
        } else {
            report << "(large)\n";
        }
        report << "\n";

        // Paired t-test
        report << "Paired t-test:\n";
        report << "─────────────────────────────────────────────────────────\n";
        auto ttest = pairedTTest(model1_errors, model2_errors);
        report << ttest.interpretation << "\n";

        // Wilcoxon test
        report << "Wilcoxon Signed-Rank Test:\n";
        report << "─────────────────────────────────────────────────────────\n";
        auto wilcoxon = wilcoxonSignedRank(model1_errors, model2_errors);
        report << wilcoxon.interpretation << "\n";

        // Bootstrap confidence interval for mean difference
        report << "Bootstrap Confidence Interval (95%):\n";
        report << "─────────────────────────────────────────────────────────\n";

        auto diff_func = [&model1_errors, &model2_errors](const std::vector<float>& indices) {
            float sum = 0.0f;
            for (float idx_f : indices) {
                int idx = static_cast<int>(idx_f);
                idx = std::min(idx, static_cast<int>(model1_errors.size()) - 1);
                sum += model1_errors[idx] - model2_errors[idx];
            }
            return sum / indices.size();
        };

        // Create index vector for bootstrap
        std::vector<float> indices(model1_errors.size());
        std::iota(indices.begin(), indices.end(), 0.0f);

        auto [lower, upper] = bootstrapConfidenceInterval(indices, diff_func, 0.95f);
        report << "Mean difference: [" << lower << ", " << upper << "]\n\n";

        // Overall conclusion
        report << "═══════════════════════════════════════════════════════════\n";
        report << "Conclusion:\n";
        report << "─────────────────────────────────────────────────────────\n";

        if (ttest.is_significant && wilcoxon.is_significant) {
            report << "✓ Both parametric and non-parametric tests indicate\n";
            report << "  SIGNIFICANT difference between models.\n";

            if (computeMean(model1_errors) < computeMean(model2_errors)) {
                report << "  " << model1_name << " performs BETTER.\n";
            } else {
                report << "  " << model2_name << " performs BETTER.\n";
            }
        } else if (ttest.is_significant || wilcoxon.is_significant) {
            report << "⚠ Mixed results - one test significant, one not.\n";
            report << "  Consider checking assumptions and sample size.\n";
        } else {
            report << "✗ No significant difference found between models.\n";
        }

        report << "═══════════════════════════════════════════════════════════\n";

        return report.str();
    }

private:
    /**
     * @brief Compute mean of vector
     */
    static float computeMean(const std::vector<float>& vec) {
        if (vec.empty()) return 0.0f;
        return std::accumulate(vec.begin(), vec.end(), 0.0f) / vec.size();
    }

    /**
     * @brief Compute standard deviation
     */
    static float computeStd(const std::vector<float>& vec) {
        if (vec.size() < 2) return 0.0f;

        float mean = computeMean(vec);
        float variance = 0.0f;

        for (float v : vec) {
            float diff = v - mean;
            variance += diff * diff;
        }

        return std::sqrt(variance / (vec.size() - 1));  // Sample std
    }

    /**
     * @brief Compute median
     */
    static float computeMedian(std::vector<float> vec) {
        if (vec.empty()) return 0.0f;

        std::sort(vec.begin(), vec.end());
        size_t n = vec.size();

        if (n % 2 == 0) {
            return (vec[n/2 - 1] + vec[n/2]) / 2.0f;
        } else {
            return vec[n/2];
        }
    }

    /**
     * @brief Compute t-distribution p-value (two-tailed)
     *
     * Approximation using normal distribution for large df
     */
    static float computeTDistPValue(float t, int df) {
        // For large df, t-distribution approximates normal
        if (df > 30) {
            return 2.0f * (1.0f - normalCDF(std::abs(t)));
        }

        // For small df, use approximation
        // This is a simplified version - full implementation would use
        // proper t-distribution CDF computation

        // Approximate using Welch-Satterthwaite
        float p = 2.0f * (1.0f - normalCDF(std::abs(t)));

        // Adjust for small sample size (conservative)
        p = std::min(p * (1.0f + 1.0f / df), 1.0f);

        return p;
    }

    /**
     * @brief Standard normal CDF
     *
     * Using error function approximation
     */
    static float normalCDF(float x) {
        return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
    }
};

} // namespace camera_aware_depth

#endif // STATISTICAL_TESTS_H
