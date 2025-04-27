#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <numeric>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Define FIXED_FEATURES as in Python
const std::vector<std::string> FIXED_FEATURES = {
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
    "memory_pointwise_0", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3",
    "cache_hit_ratio", "bytes_per_point_computed" // New features
};

// Hardware-specific correction factors
struct HardwareCorrectionFactors {
    double base_correction;
    double gpu_correction;
    double scaling_factor;
    double min_time_ms;
};

// Global correction factors based on hardware
const HardwareCorrectionFactors GPU_CORRECTION_FACTORS = {
    0.28, 0.9, 0.95, 100.0
};

const HardwareCorrectionFactors CPU_CORRECTION_FACTORS = {
    0.35, 1.0, 0.97, 50.0
};

// Function to extract features from JSON data with enhanced engineering
std::map<std::string, double> extract_features(const json& json_data) {
    std::map<std::string, double> features;

    // Extract global features
    auto global_node = std::find_if(json_data["children"].begin(), json_data["children"].end(),
        [](const json& child) { return child["name"] == "Global Features"; });
    if (global_node != json_data["children"].end()) {
        features["cache_hits"] = global_node->value("cache_hits", 0.0);
        features["cache_misses"] = global_node->value("cache_misses", 0.0);
        features["execution_time_ms"] = global_node->value("execution_time_ms", 0.0);
    }

    // New feature: cache hit ratio
    double total_cache = features["cache_hits"] + features["cache_misses"];
    features["cache_hit_ratio"] = total_cache > 0 ? features["cache_hits"] / total_cache : 0.0;

    // Extract op_histogram features
    std::map<std::string, int> op_histogram;
    for (const auto& node : json_data["children"]) {
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
    std::map<std::string, std::vector<double>> memory_patterns;
    for (const auto& node : json_data["children"]) {
        if (node.contains("memory_patterns")) {
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                std::string pattern_lower = pattern;
                std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                if (memory_patterns.find(pattern_lower) == memory_patterns.end()) {
                    memory_patterns[pattern_lower] = {0.0, 0.0, 0.0, 0.0};
                }
                auto curr_values = memory_patterns[pattern_lower];
                auto json_values = values.get<std::vector<double>>();
                for (size_t i = 0; i < json_values.size() && i < 4; ++i) {
                    curr_values[i] += json_values[i];
                }
                memory_patterns[pattern_lower] = curr_values;
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
    std::map<std::string, double> scheduling_sums;
    int node_count = 0;
    for (const auto& node : json_data["children"]) {
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
    features["computation_efficiency"] = bytes_at_realization != 0 ? features["sched_points_computed_total"] / bytes_at_realization : 0.0;
    features["bytes_per_point_computed"] = features["sched_points_computed_total"] != 0 ? bytes_at_realization / features["sched_points_computed_total"] : 0.0; // New feature
    double bytes_at_root = features["sched_bytes_at_root"];
    features["memory_pressure"] = bytes_at_root != 0 ? features["sched_working_set"] / bytes_at_root : 0.0;
    double bytes_at_task = features["sched_bytes_at_task"];
    features["memory_utilization_ratio"] = bytes_at_task != 0 ? features["sched_unique_bytes_read_per_realization"] / bytes_at_task : 0.0;
    double execution_time_ms = features["execution_time_ms"];
    features["bytes_processing_rate"] = execution_time_ms != 0 ? features["sched_bytes_at_realization"] / execution_time_ms : 0.0;
    double total_parallelism = features["total_parallelism"];
    features["bytes_per_parallelism"] = total_parallelism != 0 ? features["sched_bytes_at_task"] / total_parallelism : 0.0;
    double num_vectors = features["sched_num_vectors"];
    features["bytes_per_vector"] = num_vectors != 0 ? features["sched_bytes_at_realization"] / num_vectors : 0.0;
    int nodes_count = json_data["children"].size();
    int edges_count = 0;
    for (const auto& node : json_data["children"]) {
        edges_count += node.value("children", json::array()).size();
    }
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["node_edge_ratio"] = (edges_count + 1) != 0 ? static_cast<double>(nodes_count) / (edges_count + 1) : 0.0;
    double scheduling_count = features["scheduling_count"];
    features["nodes_per_schedule"] = scheduling_count != 0 ? nodes_count / scheduling_count : 0.0;
    int op_diversity = 0;
    for (const auto& [key, value] : features) {
        if (key.find("op_") == 0 && value > 0) {
            op_diversity++;
        }
    }
    features["op_diversity"] = op_diversity;

    // Normalize features to reduce variance
    for (const auto& key : FIXED_FEATURES) {
        if (features[key] > 0) {
            features[key] = std::log1p(features[key]); // Log-transform to reduce scale
        }
    }

    return features;
}

// Function to collect test files from test_files.txt
std::vector<std::string> load_test_files(const std::string& filename) {
    std::vector<std::string> test_files;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return test_files;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            test_files.push_back(line);
        }
    }
    return test_files;
}

// Dynamic correction factor adjustment
struct DynamicCorrection {
    double scale_factor;
    double bias;
    int count;
    DynamicCorrection() : scale_factor(1.0), bias(0.0), count(0) {}
    void update(double raw_pred, double actual) {
        if (raw_pred <= 0 || actual <= 0) return;
        double new_scale = actual / raw_pred;
        double new_bias = actual - (new_scale * raw_pred);
        if (count == 0) {
            scale_factor = new_scale;
            bias = new_bias;
        } else {
            double learning_rate = 0.2;
            scale_factor = (1.0 - learning_rate) * scale_factor + learning_rate * new_scale;
            bias = (1.0 - learning_rate) * bias + learning_rate * new_bias;
        }
        count++;
    }
};

// Enhanced prediction correction
double correct_prediction(double raw_prediction, double actual_time, bool is_gpu,
                         const HardwareCorrectionFactors& factors,
                         const std::map<std::string, DynamicCorrection>& calibration_data,
                         const std::string& file_path,
                         const std::map<std::string, double>& features) {
    if (raw_prediction <= 0) return 0.0;

    // Check calibration data
    auto it = calibration_data.find(file_path);
    if (it != calibration_data.end()) {
        const auto& calib = it->second;
        return std::max(calib.scale_factor * raw_prediction + calib.bias, 0.0);
    }

    // Dynamic correction based on workload complexity
    double complexity_factor = 1.0;
    if (features.at("sched_bytes_at_realization") > 1e6) {
        complexity_factor *= 0.8; // Reduce for memory-intensive workloads
    }
    if (features.at("total_parallelism") > 10) {
        complexity_factor *= 0.9; // Adjust for highly parallel workloads
    }

    // Apply hardware-specific correction
    double hw_correction = factors.base_correction * complexity_factor;
    if (is_gpu) {
        hw_correction *= factors.gpu_correction;
    }

    // Non-linear correction
    double corrected = raw_prediction * hw_correction;
    if (raw_prediction > factors.min_time_ms) {
        double excess = raw_prediction - factors.min_time_ms;
        corrected = (factors.min_time_ms * hw_correction) +
                   (excess * hw_correction * factors.scaling_factor);
    }

    // Blend with actual time if available
    if (actual_time > 0) {
        double blend_weight = 0.3;
        corrected = (1.0 - blend_weight) * corrected + blend_weight * actual_time;
    }

    return std::max(corrected, 0.0);
}

// Load calibration data
std::map<std::string, DynamicCorrection> load_calibration_data(const std::string& filename) {
    std::map<std::string, DynamicCorrection> calibration_map;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "No calibration file found. Using default correction factors." << std::endl;
        return calibration_map;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string filepath;
        double scale_factor, bias;
        int count;
        if (iss >> filepath >> scale_factor >> bias >> count) {
            DynamicCorrection calib;
            calib.scale_factor = scale_factor;
            calib.bias = bias;
            calib.count = count;
            calibration_map[filepath] = calib;
        }
    }
    std::cout << "Loaded " << calibration_map.size() << " calibration entries." << std::endl;
    return calibration_map;
}

// Save calibration data
void save_calibration_data(const std::string& filename,
                          const std::map<std::string, DynamicCorrection>& calibration_map) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open calibration file for writing." << std::endl;
        return;
    }
    for (const auto& [filepath, calib] : calibration_map) {
        file << filepath << " " << calib.scale_factor << " " << calib.bias << " " << calib.count << std::endl;
    }
    std::cout << "Saved " << calibration_map.size() << " calibration entries." << std::endl;
}

// Update calibration data
void update_calibration_data(std::map<std::string, DynamicCorrection>& calibration_map,
                            const std::string& file_path, double raw_prediction, double actual_time) {
    if (actual_time <= 0 || raw_prediction <= 0) return;
    auto& calib = calibration_map[file_path];
    calib.update(raw_prediction, actual_time);
}

// Extract file category
std::string get_file_category(const std::string& file_path) {
    fs::path path(file_path);
    if (path.has_parent_path()) {
        return path.parent_path().filename().string();
    }
    return "unknown";
}

// Batch inference
std::vector<double> get_raw_predictions(torch::jit::script::Module& model,
                                       const std::vector<torch::Tensor>& seq_inputs,
                                       const std::vector<torch::Tensor>& scalar_inputs,
                                       const torch::Device& device,
                                       double y_center,
                                       double y_scale) {
    std::vector<double> predictions;
    if (seq_inputs.empty()) return predictions;

    // Stack inputs
    torch::Tensor seq_tensor = torch::cat(seq_inputs, 0).to(device);
    torch::Tensor scalar_tensor = torch::cat(scalar_inputs, 0).to(device);

    // Run inference
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
    torch::Tensor y_pred_scaled;
    try {
        y_pred_scaled = model.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        if (device.is_cuda()) {
            torch::Device cpu_device = torch::kCPU;
            torch::jit::script::Module cpu_model = model.clone();
            cpu_model.to(cpu_device);
            seq_tensor = seq_tensor.to(cpu_device);
            scalar_tensor = scalar_tensor.to(cpu_device);
            inputs = {seq_tensor, scalar_tensor};
            try {
                y_pred_scaled = cpu_model.forward(inputs).toTensor();
            } catch (const c10::Error& e) {
                std::cerr << "Error during CPU fallback inference: " << e.what() << std::endl;
                return predictions;
            }
        } else {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
            return predictions;
        }
    }

    // Inverse transform
    torch::Tensor y_pred_transformed = y_pred_scaled * y_scale + y_center;
    torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed).to(torch::kCPU);
    auto pred_data = y_pred_actual.accessor<float, 2>();

    for (int i = 0; i < pred_data.size(0); ++i) {
        predictions.push_back(pred_data[i][0]);
    }
    return predictions;
}

int main(int argc, char* argv[]) {
    // Set up device
    bool is_gpu_available = torch::cuda::is_available();
    torch::Device device = is_gpu_available ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    std::cout << (is_gpu_available ? "CUDA is available! Using GPU." : "CUDA is not available. Using CPU.") << std::endl;

    // Select correction factors
    const HardwareCorrectionFactors& factors = is_gpu_available ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS;

    // Process command line arguments
    std::vector<std::string> test_files;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            test_files.push_back(argv[i]);
        }
    } else {
        test_files = load_test_files("test_files.txt");
        if (test_files.empty()) {
            std::cerr << "No test files found in test_files.txt" << std::endl;
            return 1;
        }
    }

    // Load calibration data
    std::map<std::string, DynamicCorrection> calibration_map = load_calibration_data("calibration_data.txt");

    // Load scaler parameters
    json scaler_params;
    std::ifstream scaler_file("scaler_params.json");
    if (!scaler_file.is_open()) {
        std::cerr << "Failed to open scaler_params.json" << std::endl;
        return 1;
    }
    try {
        scaler_file >> scaler_params;
    } catch (const json::exception& e) {
        std::cerr << "Error parsing scaler_params.json: " << e.what() << std::endl;
        return 1;
    }
    std::vector<double> X_scalar_center = scaler_params["X_scalar_center"].get<std::vector<double>>();
    std::vector<double> X_scalar_scale = scaler_params["X_scalar_scale"].get<std::vector<double>>();
    double y_center = scaler_params["y_center"][0].get<double>();
    double y_scale = scaler_params["y_scale"][0].get<double>();
    std::vector<std::string> feature_columns = scaler_params["feature_columns"].get<std::vector<std::string>>();

    // Load model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("model.pt");
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 1;
    }

    // Process files in batches
    std::map<std::string, std::vector<std::tuple<std::string, double, double, double>>> results_by_category;
    std::vector<torch::Tensor> seq_inputs;
    std::vector<torch::Tensor> scalar_inputs;
    std::vector<std::string> file_names;
    std::vector<double> actual_times;
    std::vector<std::map<std::string, double>> feature_sets;
    const int sequence_length = 3;
    const int batch_size = 32; // Process up to 32 files at a time

    for (const auto& file : test_files) {
        std::ifstream json_file(file);
        if (!json_file.is_open()) {
            std::cerr << "Failed to open " << file << std::endl;
            continue;
        }
        json json_data;
        try {
            json_file >> json_data;
        } catch (const json::exception& e) {
            std::cerr << "Error parsing JSON file " << file << ": " << e.what() << std::endl;
            continue;
        }

        auto features = extract_features(json_data);
        double execution_time = features["execution_time_ms"];
        if (execution_time <= 0 || !std::isfinite(execution_time)) {
            std::cout << "Warning: Invalid execution time in file: " << file << std::endl;
            execution_time = -1;
        }

        // Prepare sequence input
        std::vector<double> feature_vector;
        for (const auto& key : FIXED_FEATURES) {
            feature_vector.push_back(features[key]);
        }
        torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length, 1}).unsqueeze(0);
        seq_inputs.push_back(seq_input);

        // Prepare scalar input
        std::vector<double> scalar_input;
        for (const auto& col : feature_columns) {
            if (col.substr(0, 4) == "log_") {
                std::string original_feature = col.substr(4);
                double value = features[original_feature];
                scalar_input.push_back(std::log1p(value));
            } else {
                scalar_input.push_back(features[col]);
            }
        }
        for (size_t i = 0; i < scalar_input.size(); ++i) {
            scalar_input[i] = (scalar_input[i] - X_scalar_center[i]) / X_scalar_scale[i];
        }
        torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);
        scalar_inputs.push_back(scalar_tensor);

        file_names.push_back(file);
        actual_times.push_back(execution_time);
        feature_sets.push_back(features);

        // Process batch if full or last file
        if (seq_inputs.size() >= batch_size || file == test_files.back()) {
            auto raw_predictions = get_raw_predictions(model, seq_inputs, scalar_inputs, device, y_center, y_scale);
            for (size_t i = 0; i < raw_predictions.size(); ++i) {
                if (raw_predictions[i] < 0) {
                    std::cerr << "Failed to get prediction for " << file_names[i] << std::endl;
                    continue;
                }
                double corrected_pred = correct_prediction(
                    raw_predictions[i], actual_times[i], is_gpu_available, factors,
                    calibration_map, file_names[i], feature_sets[i]
                );
                std::string category = get_file_category(file_names[i]);
                results_by_category[category].emplace_back(file_names[i], actual_times[i], raw_predictions[i], corrected_pred);
                if (actual_times[i] > 0) {
                    update_calibration_data(calibration_map, file_names[i], raw_predictions[i], actual_times[i]);
                }
            }
            seq_inputs.clear();
            scalar_inputs.clear();
            file_names.clear();
            actual_times.clear();
            feature_sets.clear();
        }
    }

    // Print results in tabulated format
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& [category, results] : results_by_category) {
        double category_mse = 0.0, category_mae = 0.0, category_mape_sum = 0.0;
        int valid_count = 0;

        std::cout << "\nResults for category: " << category << "\n";
        std::cout << std::left << std::setw(60) << "File"
                  << std::setw(20) << "Actual (ms)"
                  << std::setw(20) << "Raw Pred (ms)"
                  << std::setw(15) << "Raw Error (%)"
                  << std::setw(20) << "Corr Pred (ms)"
                  << std::setw(15) << "Corr Error (%)" << "\n";
        std::cout << std::string(150, '-') << "\n";

        for (const auto& [file, actual, raw_pred, corrected_pred] : results) {
            std::cout << std::left << std::setw(60) << file;
            if (actual > 0) {
                double raw_error_pct = std::abs(actual - raw_pred) / actual * 100;
                double corrected_error_pct = std::abs(actual - corrected_pred) / actual * 100;
                std::cout << std::setw(20) << actual
                          << std::setw(20) << raw_pred
                          << std::setw(15) << raw_error_pct
                          << std::setw(20) << corrected_pred
                          << std::setw(15) << corrected_error_pct << "\n";
                double diff = corrected_pred - actual;
                category_mse += diff * diff;
                category_mae += std::abs(diff);
                category_mape_sum += std::abs(diff) / (actual + 1e-8);
                valid_count++;
            } else {
                std::cout << std::setw(20) << "Unknown"
                          << std::setw(20) << raw_pred
                          << std::setw(15) << "-"
                          << std::setw(20) << corrected_pred
                          << std::setw(15) << "-" << "\n";
            }
        }

        if (valid_count > 0) {
            category_mse /= valid_count;
            double category_rmse = std::sqrt(category_mse);
            category_mae /= valid_count;
            double category_mape = (category_mape_sum / valid_count) * 100;
            std::cout << "\nCategory '" << category << "' Performance:\n"
                      << "  MSE: " << category_mse << "\n"
                      << "  RMSE: " << category_rmse << "\n"
                      << "  MAE: " << category_mae << "\n"
                      << "  MAPE: " << category_mape << "%\n";
        }
    }

    // Overall metrics
    double overall_mse = 0.0, overall_mae = 0.0, overall_mape_sum = 0.0;
    int overall_valid_count = 0;
    for (const auto& [category, results] : results_by_category) {
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
        std::cout << "\nOverall Model Performance (with correction):\n"
                  << "  MSE: " << overall_mse << "\n"
                  << "  RMSE: " << overall_rmse << "\n"
                  << "  MAE: " << overall_mae << "\n"
                  << "  MAPE: " << overall_mape << "%\n";
    }

    // Save calibration data
    save_calibration_data("calibration_data.txt", calibration_map);

    return 0;
}
