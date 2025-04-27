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
    "memory_pointwise_0", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3"
};

// Function to extract features from JSON data
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
    features["computation_efficiency"] = (bytes_at_realization != 0) ? features["sched_points_computed_total"] / bytes_at_realization : 0.0;
    double bytes_at_root = features["sched_bytes_at_root"];
    features["memory_pressure"] = (bytes_at_root != 0) ? features["sched_working_set"] / bytes_at_root : 0.0;
    double bytes_at_task = features["sched_bytes_at_task"];
    features["memory_utilization_ratio"] = (bytes_at_task != 0) ? features["sched_unique_bytes_read_per_realization"] / bytes_at_task : 0.0;
    double execution_time_ms = features["execution_time_ms"];
    features["bytes_processing_rate"] = (execution_time_ms != 0) ? features["sched_bytes_at_realization"] / execution_time_ms : 0.0;
    double total_parallelism = features["total_parallelism"];
    features["bytes_per_parallelism"] = (total_parallelism != 0) ? features["sched_bytes_at_task"] / total_parallelism : 0.0;
    double num_vectors = features["sched_num_vectors"];
    features["bytes_per_vector"] = (num_vectors != 0) ? features["sched_bytes_at_realization"] / num_vectors : 0.0;
    int nodes_count = json_data["children"].size();
    int edges_count = 0;
    for (const auto& node : json_data["children"]) {
        edges_count += node.value("children", json::array()).size();
    }
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["node_edge_ratio"] = (edges_count + 1) != 0 ? static_cast<double>(nodes_count) / (edges_count + 1) : 0.0;
    double scheduling_count = features["scheduling_count"];
    features["nodes_per_schedule"] = (scheduling_count != 0) ? nodes_count / scheduling_count : 0.0;
    int op_diversity = 0;
    for (const auto& [key, value] : features) {
        if (key.find("op_") == 0 && value > 0) {
            op_diversity++;
        }
    }
    features["op_diversity"] = op_diversity;

    return features;
}

// Function to collect test files from test_files.txt
std::vector<std::string> load_test_files(const std::string& filename) {
    std::vector<std::string> test_files;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            test_files.push_back(line);
        }
    }
    return test_files;
}

int main() {
    // Load scaler parameters
    json scaler_params;
    std::ifstream scaler_file("scaler_params.json");
    if (!scaler_file.is_open()) {
        std::cerr << "Failed to open scaler_params.json" << std::endl;
        return 1;
    }
    scaler_file >> scaler_params;
    std::vector<double> X_scalar_center = scaler_params["X_scalar_center"].get<std::vector<double>>();
    std::vector<double> X_scalar_scale = scaler_params["X_scalar_scale"].get<std::vector<double>>();
    double y_center = scaler_params["y_center"][0].get<double>();
    double y_scale = scaler_params["y_scale"][0].get<double>();
    std::vector<std::string> feature_columns = scaler_params["feature_columns"].get<std::vector<std::string>>();

    // Load test files from test_files.txt
    std::vector<std::string> test_files = load_test_files("test_files.txt");
    if (test_files.empty()) {
        std::cerr << "No test files found in test_files.txt" << std::endl;
        return 1;
    }

    // Prepare inputs
    std::vector<torch::Tensor> seq_inputs;
    std::vector<torch::Tensor> scalar_inputs;
    std::vector<double> actual_times;
    std::vector<std::string> file_names;
    const int sequence_length = 3;

    for (const auto& file : test_files) {
        std::ifstream json_file(file);
        if (!json_file.is_open()) {
            std::cerr << "Failed to open " << file << std::endl;
            continue;
        }
        json json_data;
        json_file >> json_data;
        auto features = extract_features(json_data);

        // Skip invalid execution times
        double execution_time = features["execution_time_ms"];
        if (execution_time <= 0 || !std::isfinite(execution_time)) {
            std::cout << "Skipped file with invalid execution time: " << file << std::endl;
            continue;
        }
        actual_times.push_back(execution_time);
        file_names.push_back(file);

        // Sequence input
        std::vector<double> feature_vector;
        for (const auto& key : FIXED_FEATURES) {
            feature_vector.push_back(features[key]);
        }
        torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length, 1});
        seq_inputs.push_back(seq_input.unsqueeze(0));

        // Scalar input
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
        // Scale scalar input
        for (size_t i = 0; i < scalar_input.size(); ++i) {
            scalar_input[i] = (scalar_input[i] - X_scalar_center[i]) / X_scalar_scale[i];
        }
        torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32);
        scalar_inputs.push_back(scalar_tensor.unsqueeze(0));
    }

    if (seq_inputs.empty()) {
        std::cerr << "No valid test data processed" << std::endl;
        return 1;
    }

    // Stack into batched tensors
    torch::Tensor seq_inputs_tensor = torch::cat(seq_inputs, 0);
    torch::Tensor scalar_inputs_tensor = torch::cat(scalar_inputs, 0);

    // Load the model with the correct type
    std::shared_ptr<torch::jit::script::Module> model;
    try {
        model = torch::jit::load("model.pt");
        model->eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 1;
    }

    // Run inference
    std::vector<torch::jit::IValue> inputs = {seq_inputs_tensor, scalar_inputs_tensor};
    torch::Tensor y_pred_scaled = model->forward(inputs).toTensor();

    // Inverse transform predictions
    torch::Tensor y_pred_transformed = y_pred_scaled * y_scale + y_center;
    torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);

    // Convert to CPU for evaluation
    y_pred_actual = y_pred_actual.to(torch::kCPU);
    auto pred_data = y_pred_actual.accessor<float, 2>();

    // Group results by subfolder and compute metrics
    std::map<std::string, std::vector<std::tuple<std::string, double, double>>> results_by_subfolder;
    std::vector<double> all_preds, all_actuals;
    for (size_t i = 0; i < actual_times.size(); ++i) {
        std::string file_path = file_names[i];
        std::string subfolder = file_path.substr(0, file_path.find_last_of('/'));
        double actual = actual_times[i];
        double pred = std::max(static_cast<double>(pred_data[i][0]), 0.0);
        all_preds.push_back(pred);
        all_actuals.push_back(actual);
        results_by_subfolder[subfolder].emplace_back(file_path, actual, pred);
    }

    // Print results by subfolder
    for (const auto& [subfolder, results] : results_by_subfolder) {
        std::cout << "\nResults for " << subfolder << ":\n";
        for (const auto& [file, actual, pred] : results) {
            double error_percentage = (actual > 0) ? std::abs(actual - pred) / actual * 100 : 0;
            std::cout << "File: " << file << "\n";
            std::cout << "  Actual execution time: " << actual << " ms\n";
            std::cout << "  Predicted execution time: " << pred << " ms\n";
            std::cout << "  Error percentage: " << error_percentage << "%\n";
        }
    }

    // Compute overall metrics
    double mse = 0.0, mae = 0.0, mape_sum = 0.0;
    for (size_t i = 0; i < all_preds.size(); ++i) {
        double diff = all_preds[i] - all_actuals[i];
        mse += diff * diff;
        mae += std::abs(diff);
        mape_sum += (all_actuals[i] > 0) ? std::abs(diff) / (all_actuals[i] + 1e-8) : 0;
    }
    mse /= all_preds.size();
    double rmse = std::sqrt(mse);
    mae /= all_preds.size();
    double mape = (mape_sum / all_preds.size()) * 100;

    std::cout << "\nOverall Model Performance:\n";
    std::cout << "MSE: " << mse << "\n";
    std::cout << "RMSE: " << rmse << "\n";
    std::cout << "MAE: " << mae << "\n";
    std::cout << "MAPE: " << mape << "%\n";

    return 0;
}
