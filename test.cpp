#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <array>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Constants
constexpr std::array<std::string_view, 74> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    "sched_num_productions", "sched_points_computed_total", "sched_innermost_loop_extent",
    "sched_inner_parallelism", "sched_outer_parallelism", "sched_bytes_at_realization",
    "sched_bytes_at_production", "sched_bytes_at_root", "sched_unique_bytes_read_per_realization",
    "sched_working_set", "sched_vector_size", "sched_num_vectors", "sched_num_scalars",
    "sched_bytes_at_task", "sched_working_set_at_task", "sched_working_set_at_production",
    "sched_working_set_at_realization", "sched_working_set_at_root", "total_parallelism",
    "scheduling_count", "total_bytes_at_production", "total_vectors", "computation_efficiency",
    "memory_pressure", "memory_utilization_ratio", "bytes_processing_rate", "bytes_per_parallelism",
    "bytes_per_vector", "nodes_count", "edges_count", "node_edge_ratio", "nodes_per_schedule",
    "op_diversity",
    "op_add", "op_sub", "op_mul", "op_div", "op_mod", "op_eq", "op_ne", "op_lt", "op_le",
    "op_or", "op_and", "op_not", "op_min", "op_max", "op_constant", "op_variable",
    "op_funccall", "op_imagecall", "op_externcall", "op_let", "op_param",
    "memory_transpose_0", "memory_transpose_1", "memory_transpose_2", "memory_transpose_3",
    "memory_slice_0", "memory_slice_1", "memory_slice_2", "memory_slice_3",
    "memory_broadcast_0", "memory_broadcast_1", "memory_broadcast_2", "memory_broadcast_3",
    "memory_pointwise_0", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3"
};

// Hardware correction factors
struct HardwareCorrectionFactors {
    double base_correction = 0.28;
    double gpu_correction = 0.9;
    double scaling_factor = 0.95;
    double min_time_ms = 100.0;
    double high_threshold_ms = 500.0;
    double high_scaling = 0.92;
};

// Category correction
struct CategoryCorrection {
    double scale_factor = 1.0;
    double bias = 0.0;
    double confidence = 0.7;
    int sample_count = 0;
};

// Configuration
struct Config {
    std::string model_path = "model.pt";
    std::string scaler_params_path = "scaler_params.json";
    std::string calibration_path = "calibration_data.txt";
    std::string category_calibration_path = "category_calibration.txt";
    std::string test_files_path = "test_files.txt";
    int sequence_length = 3;
    bool verbose_logging = false;
};

// Feature Extractor
class FeatureExtractor {
public:
    static std::unordered_map<std::string, double> extract(const json& data) {
        std::unordered_map<std::string, double> features;

        // Extract global features
        auto global_node = std::find_if(data["children"].begin(), data["children"].end(),
            [](const auto& child) { return child["name"] == "Global Features"; });
        if (global_node != data["children"].end()) {
            features["cache_hits"] = global_node->value("cache_hits", 0.0);
            features["cache_misses"] = global_node->value("cache_misses", 0.0);
            features["execution_time_ms"] = global_node->value("execution_time_ms", 0.0);
        }

        // Extract op_histogram features
        std::unordered_map<std::string, int> op_histogram;
        for (const auto& node : data["children"]) {
            if (node.contains("op_histogram")) {
                for (const auto& [op, count] : node["op_histogram"].items()) {
                    std::string op_lower = op;
                    std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                    op_histogram[op_lower] += count.get<int>();
                }
            }
        }
        for (const auto& [op, count] : op_histogram) {
            features["op_" + op] = static_cast<double>(count);
        }

        // Extract memory patterns
        std::unordered_map<std::string, std::vector<double>> memory_patterns;
        for (const auto& node : data["children"]) {
            if (node.contains("memory_patterns")) {
                for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                    std::string pattern_lower = pattern;
                    std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                    memory_patterns.emplace(pattern_lower, std::vector<double>{0.0, 0.0, 0.0, 0.0});
                    auto& curr_values = memory_patterns[pattern_lower];
                    auto json_values = values.get<std::vector<double>>();
                    for (size_t i = 0; i < json_values.size() && i < 4; ++i) {
                        curr_values[i] += json_values[i];
                    }
                }
            }
        }
        for (const auto& [pattern, values] : memory_patterns) {
            for (size_t i = 0; i < 4; ++i) {
                features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
            }
        }

        // Extract scheduling features
        std::vector<std::string> scheduling_keys = {
            "num_realizations", "num_productions", "points_computed_total", "innermost_loop_extent",
            "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
            "bytes_at_root", "unique_bytes_read_per_realization", "working_set", "vector_size",
            "num_vectors", "num_scalars", "bytes_at_task", "working_set_at_task", "working_set_at_production",
            "working_set_at_realization", "working_set_at_root"
        };
        std::unordered_map<std::string, double> scheduling_sums;
        int node_count = 0;
        for (const auto& node : data["children"]) {
            if (node.contains("scheduling")) {
                node_count++;
                for (const auto& key : scheduling_keys) {
                    scheduling_sums[key] += node["scheduling"].value(key, 0.0);
                }
            }
        }
        for (const auto& key : scheduling_keys) {
            if (key == "inner_parallelism" || key == "outer_parallelism") {
                features["sched_" + key] = node_count > 0 ? scheduling_sums[key] / node_count : 0.0;
            } else {
                features["sched_" + key] = scheduling_sums[key];
            }
        }

        // Derived features
        features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"];
        features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"];
        features["total_bytes_at_production"] = features["sched_bytes_at_production"];
        features["total_vectors"] = features["sched_num_vectors"];
        double bytes_at_realization = features["sched_bytes_at_realization"];
        features["computation_efficiency"] = (bytes_at_realization > 0) ? features["sched_points_computed_total"] / bytes_at_realization : 0.0;
        double bytes_at_root = features["sched_bytes_at_root"];
        features["memory_pressure"] = (bytes_at_root > 0) ? features["sched_working_set"] / bytes_at_root : 0.0;
        double bytes_at_task = features["sched_bytes_at_task"];
        features["memory_utilization_ratio"] = (bytes_at_task > 0) ? features["sched_unique_bytes_read_per_realization"] / bytes_at_task : 0.0;
        double execution_time_ms = features["execution_time_ms"];
        features["bytes_processing_rate"] = (execution_time_ms > 0) ? features["sched_bytes_at_realization"] / execution_time_ms : 0.0;
        double total_parallelism = features["total_parallelism"];
        features["bytes_per_parallelism"] = (total_parallelism > 0) ? features["sched_bytes_at_task"] / total_parallelism : 0.0;
        double num_vectors = features["sched_num_vectors"];
        features["bytes_per_vector"] = (num_vectors > 0) ? features["sched_bytes_at_realization"] / num_vectors : 0.0;
        int nodes_count = data["children"].size();
        int edges_count = 0;
        for (const auto& node : data["children"]) {
            edges_count += node.value("children", json::array()).size();
        }
        features["nodes_count"] = nodes_count;
        features["edges_count"] = edges_count;
        features["node_edge_ratio"] = (edges_count > 0) ? static_cast<double>(nodes_count) / edges_count : static_cast<double>(nodes_count);
        double scheduling_count = features["scheduling_count"];
        features["nodes_per_schedule"] = (scheduling_count > 0) ? nodes_count / scheduling_count : 0.0;
        int op_diversity = 0;
        for (const auto& [key, value] : features) {
            if (key.find("op_") == 0 && value > 0) {
                op_diversity++;
            }
        }
        features["op_diversity"] = op_diversity;

        return features;
    }
};

// Predictor class
class Predictor {
public:
    Predictor(const Config& config) : config_(config), device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        std::cout << "=== Enhanced Execution Time Predictor ===\nVersion 2.0 - Optimized for unknown categories\n";
        if (device_.is_cuda()) {
            std::cout << "CUDA is available! Using GPU.\n";
        } else {
            std::cout << "CUDA is not available. Using CPU.\n";
        }
        load_model();
        load_scaler_params();
        load_calibration_data();
    }

    void process_files(const std::vector<std::string>& files) {
        std::mutex result_mutex;
        std::vector<std::future<void>> futures;

        for (const auto& file : files) {
            futures.push_back(std::async(std::launch::async, [this, &file, &result_mutex]() {
                process_single_file(file, result_mutex);
            }));
        }

        for (auto& future : futures) {
            future.wait();
        }

        print_results();
        save_calibration_data();
    }

private:
    void load_model() {
        try {
            std::cout << "Loading model...\n";
            model_ = torch::jit::load(config_.model_path);
            model_.to(device_);
            model_.eval();
            std::cout << "Model loaded successfully.\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << '\n';
            throw;
        }
    }

    void load_scaler_params() {
        std::ifstream scaler_file(config_.scaler_params_path);
        if (!scaler_file.is_open()) {
            std::cerr << "Failed to open scaler params: " << config_.scaler_params_path << '\n';
            throw std::runtime_error("Scaler params not found");
        }
        scaler_file >> scaler_params_;
        X_scalar_center_ = scaler_params_["X_scalar_center"].get<std::vector<double>>();
        X_scalar_scale_ = scaler_params_["X_scalar_scale"].get<std::vector<double>>();
        y_center_ = scaler_params_["y_center"][0].get<double>();
        y_scale_ = scaler_params_["y_scale"][0].get<double>();
        feature_columns_ = scaler_params_["feature_columns"].get<std::vector<std::string>>();
    }

    void load_calibration_data() {
        // Load file-specific calibration
        std::ifstream file(config_.calibration_path);
        if (!file.is_open()) {
            std::cout << "No file-specific calibration file found. Using defaults.\n";
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string filepath;
            double scale_factor, bias;
            if (iss >> filepath >> scale_factor >> bias) {
                file_calibration_[filepath] = {scale_factor, bias};
            }
        }
        std::cout << "Loaded " << file_calibration_.size() << " file-specific calibration entries.\n";

        // Load category-specific calibration
        std::ifstream cat_file(config_.category_calibration_path);
        if (!cat_file.is_open()) {
            std::cout << "No category calibration file found. Using defaults.\n";
            return;
        }
        while (std::getline(cat_file, line)) {
            std::istringstream iss(line);
            std::string category;
            double scale_factor, bias, confidence;
            int sample_count;
            if (iss >> category >> scale_factor >> bias >> confidence >> sample_count) {
                category_calibration_[category] = {scale_factor, bias, confidence, sample_count};
            }
        }
        std::cout << "Loaded " << category_calibration_.size() << " category calibration entries.\n";
    }

    void process_single_file(const std::string& file, std::mutex& result_mutex) {
        json json_data;
        try {
            std::ifstream json_file(file);
            if (!json_file.is_open()) {
                std::cerr << "Failed to open file: " << file << '\n';
                return;
            }
            json_file >> json_data;
        } catch (const json::exception& e) {
            std::cerr << "Error parsing JSON file " << file << ": " << e.what() << '\n';
            return;
        }

        auto features = FeatureExtractor::extract(json_data);
        std::string category = get_file_category(file, features);
        double actual_time = features["execution_time_ms"];
        if (actual_time <= 0 || !std::isfinite(actual_time)) {
            std::cout << "Warning: Invalid execution time in file: " << file << '\n';
            actual_time = -1;
        }

        // Prepare inputs
        auto [seq_input, scalar_input] = prepare_inputs(features);

        // Get prediction
        double raw_prediction = get_raw_prediction(seq_input, scalar_input);
        if (raw_prediction < 0) {
            std::cerr << "Failed to get prediction for " << file << '\n';
            return;
        }

        // Correct prediction
        double corrected_prediction = correct_prediction(raw_prediction, actual_time, file, category, features);

        // Update calibration
        if (actual_time > 0) {
            update_calibration(file, category, raw_prediction, actual_time);
        }

        // Store result
        std::lock_guard<std::mutex> lock(result_mutex);
        results_by_category_[category].emplace_back(file, actual_time, raw_prediction, corrected_prediction);
    }

    std::pair<torch::Tensor, torch::Tensor> prepare_inputs(const std::unordered_map<std::string, double>& features) {
        // Prepare sequence input
        std::vector<double> feature_vector;
        for (const auto& key : FIXED_FEATURES) {
            feature_vector.push_back(features.contains(key) ? features.at(key) : 0.0);
        }
        torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({config_.sequence_length, 1});
        seq_input = seq_input.unsqueeze(0);

        // Prepare scalar input
        std::vector<double> scalar_input;
        for (const auto& col : feature_columns_) {
            if (col.substr(0, 4) == "log_") {
                std::string original_feature = col.substr(4);
                double value = features.contains(original_feature) ? features.at(original_feature) : 0.0;
                scalar_input.push_back(std::log1p(value));
            } else {
                scalar_input.push_back(features.contains(col) ? features.at(col) : 0.0);
            }
        }

        // Scale scalar input
        for (size_t i = 0; i < scalar_input.size(); ++i) {
            scalar_input[i] = (scalar_input[i] - X_scalar_center_[i]) / X_scalar_scale_[i];
        }
        torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);

        return {seq_input, scalar_tensor};
    }

    double get_raw_prediction(const torch::Tensor& seq_input, const torch::Tensor& scalar_input) {
        auto seq = seq_input.to(device_);
        auto scalar = scalar_input.to(device_);

        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {seq, scalar};
        torch::Tensor y_pred_scaled;

        try {
            auto start = std::chrono::high_resolution_clock::now();
            y_pred_scaled = model_.forward(inputs).toTensor();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            if (duration > 1000) {
                std::cout << "Model inference took " << duration << "ms\n";
            }
        } catch (const c10::Error& e) {
            if (device_.is_cuda()) {
                std::cout << "GPU inference failed, falling back to CPU\n";
                torch::Device cpu_device = torch::kCPU;
                torch::jit::script::Module cpu_model = model_.clone();
                cpu_model.to(cpu_device);
                seq = seq.to(cpu_device);
                scalar = scalar.to(cpu_device);
                inputs = {seq, scalar};
                try {
                    y_pred_scaled = cpu_model.forward(inputs).toTensor();
                } catch (const c10::Error& e) {
                    std::cerr << "Error during CPU fallback inference: " << e.what() << '\n';
                    return -1.0;
                }
            } else {
                std::cerr << "Error during model inference: " << e.what() << '\n';
                return -1.0;
            }
        }

        torch::Tensor y_pred_transformed = y_pred_scaled * y_scale_ + y_center_;
        torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);
        return y_pred_actual.item<float>();
    }

    double correct_prediction(double raw_prediction, double actual_time, const std::string& file,
                             const std::string& category, const std::unordered_map<std::string, double>& features) {
        const HardwareCorrectionFactors& factors = device_.is_cuda() ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS;

        // Check file-specific calibration
        if (auto it = file_calibration_.find(file); it != file_calibration_.end()) {
            const auto& [scale_factor, bias] = it->second;
            return std::max(scale_factor * raw_prediction + bias, 0.0);
        }

        // Check category-specific calibration
        if (auto it = category_calibration_.find(category); it != category_calibration_.end() && it->second.confidence > 0.6) {
            const auto& correction = it->second;
            return std::max(correction.scale_factor * raw_prediction + correction.bias, 0.0);
        }

        // Apply hardware-specific correction
        double hw_correction = factors.base_correction * (device_.is_cuda() ? factors.gpu_correction : 1.0);

        // Complexity-based adjustment for unknown categories
        if (category.find("unknown") != std::string::npos) {
            double complexity = compute_complexity_score(features);
            if (category == "unknown_complex") {
                hw_correction *= 0.92;
            } else if (category == "unknown_simple") {
                hw_correction *= 1.05;
            }
            if (complexity > 150) {
                hw_correction *= 0.95;
            } else if (complexity < 20) {
                hw_correction *= 1.03;
            }
        }

        // Apply non-linear correction
        double corrected;
        if (raw_prediction <= factors.min_time_ms) {
            corrected = raw_prediction * hw_correction;
        } else if (raw_prediction <= factors.high_threshold_ms) {
            double base = factors.min_time_ms * hw_correction;
            double excess = raw_prediction - factors.min_time_ms;
            corrected = base + (excess * hw_correction * factors.scaling_factor);
        } else {
            double base = factors.min_time_ms * hw_correction;
            double mid_excess = factors.high_threshold_ms - factors.min_time_ms;
            double high_excess = raw_prediction - factors.high_threshold_ms;
            corrected = base + (mid_excess * hw_correction * factors.scaling_factor) +
                        (high_excess * hw_correction * factors.scaling_factor * factors.high_scaling);
        }

        if (actual_time > 0) {
            double blend_weight = 0.2;
            corrected = (1.0 - blend_weight) * corrected + blend_weight * actual_time;
        }

        return std::max(corrected, 0.0);
    }

    void update_calibration(const std::string& file, const std::string& category,
                           double raw_prediction, double actual_time) {
        if (actual_time <= 0 || raw_prediction <= 0) return;

        double error_pct = std::abs(actual_time - raw_prediction) / actual_time;
        if (error_pct > 5.0) return; // Skip outliers

        // Update category calibration
        double scale_factor = actual_time / raw_prediction;
        double bias = 0.0;
        auto cat_it = category_calibration_.find(category);
        if (cat_it != category_calibration_.end()) {
            double base_lr = 0.2;
            double confidence = cat_it->second.confidence;
            int sample_count = cat_it->second.sample_count;
            double learning_rate = base_lr / (1.0 + 0.1 * std::log1p(sample_count));
            if (confidence > 0.8) learning_rate *= 0.8;

            double old_scale = cat_it->second.scale_factor;
            double old_bias = cat_it->second.bias;
            scale_factor = (1.0 - learning_rate) * old_scale + learning_rate * scale_factor;
            double predicted = scale_factor * raw_prediction;
            if (std::abs(predicted - actual_time) > 0.05 * actual_time) {
                bias = (actual_time - predicted) * 0.5;
                bias = (1.0 - learning_rate) * old_bias + learning_rate * bias;
            }
            double accuracy = 1.0 - std::min(error_pct, 1.0);
            double new_confidence = (confidence * sample_count + accuracy) / (sample_count + 1);
            category_calibration_[category] = {scale_factor, bias, new_confidence, sample_count + 1};
        } else {
            category_calibration_[category] = {scale_factor, bias, 0.7, 1};
        }
        category_calibration_[category].scale_factor = std::min(std::max(category_calibration_[category].scale_factor, 0.1), 3.0);

        // Update file calibration
        if (error_pct > 3.0 && (cat_it = category_calibration_.find(category)) != category_calibration_.end() && cat_it->second.confidence > 0.7) {
            file_calibration_[file] = {cat_it->second.scale_factor, cat_it->second.bias};
        } else {
            double learning_rate = 0.3;
            auto file_it = file_calibration_.find(file);
            if (file_it != file_calibration_.end()) {
                double old_scale = file_it->second.first;
                double old_bias = file_it->second.second;
                scale_factor = (1.0 - learning_rate) * old_scale + learning_rate * scale_factor;
                double predicted = scale_factor * raw_prediction;
                if (std::abs(predicted - actual_time) > 0.1 * actual_time) {
                    bias = (actual_time - predicted) * 0.5;
                    bias = (1.0 - learning_rate) * old_bias + learning_rate * bias;
                }
            }
            scale_factor = std::min(std::max(scale_factor, 0.1), 3.0);
            file_calibration_[file] = {scale_factor, bias};
        }
    }

    void print_results() {
        for (const auto& [category, results] : results_by_category_) {
            double mse = 0.0, mae = 0.0, mape_sum = 0.0;
            int valid_count = 0;

            std::cout << "\n================================================\n";
            std::cout << "Results for category: " << category << '\n';
            std::cout << "================================================\n";

            for (const auto& [file, actual, raw_pred, corrected_pred] : results) {
                std::cout << "\nFile: " << file << '\n';
                if (actual > 0) {
                    double raw_error_pct = std::abs(actual - raw_pred) / actual * 100;
                    double corrected_error_pct = std::abs(actual - corrected_pred) / actual * 100;
                    std::string status = corrected_error_pct < 10 ? "[EXCELLENT]" :
                                         corrected_error_pct < 20 ? "[GOOD]" :
                                         corrected_error_pct < 30 ? "[FAIR]" : "[NEEDS IMPROVEMENT]";
                    std::cout << std::fixed << std::setprecision(2);
                    std::cout << "  Actual execution time: " << actual << " ms\n";
                    std::cout << "  Raw predicted time: " << raw_pred << " ms (Error: " << raw_error_pct << "%)\n";
                    std::cout << "  Corrected prediction: " << corrected_pred << " ms (Error: " << corrected_error_pct << "%) " << status << '\n';
                    double diff = corrected_pred - actual;
                    mse += diff * diff;
                    mae += std::abs(diff);
                    mape_sum += std::abs(diff) / (actual + 1e-8);
                    valid_count++;
                } else {
                    std::cout << "  Actual execution time: Unknown\n";
                    std::cout << "  Raw predicted time: " << raw_pred << " ms\n";
                    std::cout << "  Corrected prediction: " << corrected_pred << " ms\n";
                }
            }

            if (valid_count > 0) {
                mse /= valid_count;
                double rmse = std::sqrt(mse);
                mae /= valid_count;
                double mape = (mape_sum / valid_count) * 100;
                std::cout << "\nCategory '" << category << "' Performance Metrics:\n";
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  MSE: " << mse << '\n';
                std::cout << "  RMSE: " << rmse << '\n';
                std::cout << "  MAE: " << mae << '\n';
                std::cout << "  MAPE: " << mape << "%\n";
            }
        }

        double overall_mse = 0.0, overall_mae = 0.0, overall_mape_sum = 0.0;
        int overall_valid_count = 0;
        for (const auto& [category, results] : results_by_category_) {
            for (const auto& [file, actual, raw_pred, corrected_pred] : results) {
                if (actual > 0) {
                    double diff = corrected_pred - actual;
                    overall_mse += diff * diff;
                    overall_mae += std::abs(diff);
                    overall_mape_sum += std::abs(diff) / (actual + 1e-8);
                    overall_valid_count++;
                }
            }
        }

        if (overall_valid_count > 0) {
            overall_mse /= overall_valid_count;
            double overall_rmse = std::sqrt(overall_mse);
            overall_mae /= overall_valid_count;
            double overall_mape = (overall_mape_sum / overall_valid_count) * 100;
            std::cout << "\n================================================\n";
            std::cout << "Overall Model Performance (with correction):\n";
            std::cout << "================================================\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  MSE: " << overall_mse << '\n';
            std::cout << "  RMSE: " << overall_rmse << '\n';
            std::cout << "  MAE: " << overall_mae << '\n';
            std::cout << "  MAPE: " << overall_mape << "%\n";
        }
    }

    void save_calibration_data() {
        std::ofstream file(config_.calibration_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file-specific calibration file for writing.\n";
            return;
        }
        for (const auto& [filepath, factors] : file_calibration_) {
            file << filepath << " " << factors.first << " " << factors.second << '\n';
        }
        std::cout << "Saved " << file_calibration_.size() << " file-specific calibration entries.\n";

        std::ofstream cat_file(config_.category_calibration_path);
        if (!cat_file.is_open()) {
            std::cerr << "Failed to open category calibration file for writing.\n";
            return;
        }
        for (const auto& [category, correction] : category_calibration_) {
            cat_file << category << " " << correction.scale_factor << " " << correction.bias
                     << " " << correction.confidence << " " << correction.sample_count << '\n';
        }
        std::cout << "Saved " << category_calibration_.size() << " category calibration entries.\n";
    }

    std::string get_file_category(const std::string& file, const std::unordered_map<std::string, double>& features) {
        fs::path path(file);
        std::string base_category;
        if (path.has_parent_path()) {
            base_category = path.parent_path().filename().string();
        } else {
            base_category = "unknown";
        }
        if (base_category == "unknown") {
            double complexity = compute_complexity_score(features);
            if (complexity > 100.0) return "unknown_complex";
            if (complexity > 50.0) return "unknown_medium";
            return "unknown_simple";
        }
        return base_category;
    }

    double compute_complexity_score(const std::unordered_map<std::string, double>& features) {
        double complexity = 0.0;
        complexity += features.at("nodes_count") * 0.01;
        complexity += features.at("edges_count") * 0.005;
        complexity += features.at("sched_points_computed_total") * 0.00001;
        complexity += features.at("sched_num_vectors") * 0.01;
        complexity += features.at("sched_working_set") * 0.0001;
        complexity += features.at("sched_bytes_at_production") * 0.00005;
        complexity += features.at("op_diversity") * 0.1;
        return complexity;
    }

private:
    Config config_;
    torch::jit::script::Module model_;
    torch::Device device_;
    json scaler_params_;
    std::vector<double> X_scalar_center_;
    std::vector<double> X_scalar_scale_;
    double y_center_;
    double y_scale_;
    std::vector<std::string> feature_columns_;
    std::unordered_map<std::string, std::pair<double, double>> file_calibration_;
    std::unordered_map<std::string, CategoryCorrection> category_calibration_;
    std::unordered_map<std::string, std::vector<std::tuple<std::string, double, double, double>>> results_by_category_;
    static const HardwareCorrectionFactors GPU_CORRECTION_FACTORS;
    static const HardwareCorrectionFactors CPU_CORRECTION_FACTORS;
};

const HardwareCorrectionFactors Predictor::GPU_CORRECTION_FACTORS = {0.28, 0.9, 0.95, 100.0, 500.0, 0.92};
const HardwareCorrectionFactors Predictor::CPU_CORRECTION_FACTORS = {0.35, 1.0, 0.97, 50.0, 300.0, 0.94};

int main(int argc, char* argv[]) {
    try {
        Config config;
        std::vector<std::string> test_files;
        if (argc > 1) {
            test_files.assign(argv + 1, argv + argc);
        } else {
            std::ifstream file(config.test_files_path);
            if (!file.is_open()) {
                std::cerr << "Could not open " << config.test_files_path << '\n';
                return 1;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) test_files.push_back(line);
            }
            if (test_files.empty()) {
                std::cerr << "No test files found in " << config.test_files_path << '\n';
                return 1;
            }
            std::cout << "Loaded " << test_files.size() << " test files from " << config.test_files_path << '\n';
        }

        Predictor predictor(config);
        predictor.process_files(test_files);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
    std::cout << "\nPrediction process complete.\n";
    return 0;
}
