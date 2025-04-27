#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <filesystem>

// Using JSON namespace
using json = nlohmann::json;

// Define fixed features (same as Python)
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

// Define low-importance features to drop (same as Python)
const std::vector<std::string> LOW_IMPORTANCE_FEATURES = {
    "op_cast", "op_selfcall", "memory_pointwise_1", "memory_transpose_1", "memory_broadcast_1",
    "memory_slice_1", "op_select", "op_not", "op_and", "op_ne", "op_mod", "memory_pointwise_2",
    "memory_broadcast_2", "memory_slice_2", "memory_transpose_2", "op_externcall", "op_imagecall",
    "op_param", "memory_pointwise_3", "memory_transpose_3", "op_sub", "memory_pointwise_0", "op_let"
};

// Define skewed features for log transformation
const std::vector<std::string> SKEWED_FEATURES = {
    "cache_hits", "bytes_processing_rate", "sched_bytes_at_task", "computation_efficiency"
};

// Feature extraction function (translated from Python)
std::map<std::string, double> extract_features(const json& json_data) {
    std::map<std::string, double> features;

    // Extract global features
    for (const auto& child : json_data["children"]) {
        if (child["name"] == "Global Features") {
            features["cache_hits"] = child.value("cache_hits", 0.0);
            features["cache_misses"] = child.value("cache_misses", 0.0);
            features["execution_time_ms"] = child.value("execution_time_ms", 0.0);
            break;
        }
    }

    // Extract op_histogram features
    std::map<std::string, double> op_histogram;
    for (const auto& node : json_data["children"]) {
        if (node.contains("op_histogram")) {
            for (const auto& [op, count] : node["op_histogram"].items()) {
                std::string op_lower = op;
                std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                op_histogram[op_lower] += count.get<double>();
            }
        }
    }
    for (const auto& [op, count] : op_histogram) {
        features["op_" + op] = count;
    }

    // Extract memory patterns
    std::map<std::string, std::vector<double>> memory_patterns;
    for (const auto& node : json_data["children"]) {
        if (node.contains("memory_patterns")) {
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                std::vector<double> curr_values(4, 0.0);
                for (size_t i = 0; i < values.size() && i < 4; ++i) {
                    curr_values[i] = values[i].get<double>();
                }
                if (memory_patterns.find(pattern) == memory_patterns.end()) {
                    memory_patterns[pattern] = std::vector<double>(4, 0.0);
                }
                for (size_t i = 0; i < 4; ++i) {
                    memory_patterns[pattern][i] += curr_values[i];
                }
            }
        }
    }
    for (const auto& [pattern, values] : memory_patterns) {
        std::string pattern_lower = pattern;
        std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
        for (size_t i = 0; i < values.size(); ++i) {
            features["memory_" + pattern_lower + "_" + std::to_string(i)] = values[i];
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
            ++node_count;
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
    features["computation_efficiency"] = features["sched_bytes_at_realization"] != 0 ?
        features["sched_points_computed_total"] / features["sched_bytes_at_realization"] : 0.0;
    features["memory_pressure"] = features["sched_bytes_at_root"] != 0 ?
        features["sched_working_set"] / features["sched_bytes_at_root"] : 0.0;
    features["memory_utilization_ratio"] = features["sched_bytes_at_task"] != 0 ?
        features["sched_unique_bytes_read_per_realization"] / features["sched_bytes_at_task"] : 0.0;
    features["bytes_processing_rate"] = features["execution_time_ms"] != 0 ?
        features["sched_bytes_at_realization"] / features["execution_time_ms"] : 0.0;
    features["bytes_per_parallelism"] = features["total_parallelism"] != 0 ?
        features["sched_bytes_at_task"] / features["total_parallelism"] : 0.0;
    features["bytes_per_vector"] = features["sched_num_vectors"] != 0 ?
        features["sched_bytes_at_realization"] / features["sched_num_vectors"] : 0.0;
    double nodes_count = json_data["children"].size();
    double edges_count = 0;
    for (const auto& node : json_data["children"]) {
        edges_count += node.value("children", json::array()).size();
    }
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["node_edge_ratio"] = edges_count + 1 != 0 ? nodes_count / (edges_count + 1) : 0.0;
    features["nodes_per_schedule"] = features["scheduling_count"] != 0 ?
        nodes_count / features["scheduling_count"] : 0.0;
    features["op_diversity"] = std::count_if(features.begin(), features.end(),
        [](const auto& kv) { return kv.first.find("op_") == 0 && kv.second > 0; });

    // Create fixed-length feature vector
    std::map<std::string, double> fixed_features;
    for (const auto& key : FIXED_FEATURES) {
        fixed_features[key] = features[key];
    }
    return fixed_features;
}

// Structure to hold scaler parameters
struct ScalerParams {
    std::vector<double> x_scalar_center;
    std::vector<double> x_scalar_scale;
    std::vector<double> y_center;
    std::vector<double> y_scale;
    std::vector<std::string> feature_columns;
};

// Load scaler parameters from JSON
ScalerParams load_scaler_params(const std::string& scaler_file) {
    ScalerParams params;
    std::ifstream ifs(scaler_file);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open scaler_params.json");
    }
    json scaler_data;
    ifs >> scaler_data;
    params.x_scalar_center = scaler_data["X_scalar_center"].get<std::vector<double>>();
    params.x_scalar_scale = scaler_data["X_scalar_scale"].get<std::vector<double>>();
    params.y_center = scaler_data["y_center"].get<std::vector<double>>();
    params.y_scale = scaler_data["y_scale"].get<std::vector<double>>();
    params.feature_columns = scaler_data["feature_columns"].get<std::vector<std::string>>();
    return params;
}

// Preprocessing function
struct PreprocessedData {
    torch::Tensor seq_input;
    torch::Tensor scalar_input;
    double execution_time_ms;
};

PreprocessedData preprocess_features(const std::map<std::string, double>& features, const ScalerParams& scaler_params) {
    // Create sequence input (sequence_length=3)
    const int sequence_length = 3;
    std::vector<double> feature_vector;
    for (const auto& key : FIXED_FEATURES) {
        double val = features.at(key);
        if (!std::isfinite(val)) {
            std::cerr << "Warning: Non-finite value for feature " << key << ": " << val << std::endl;
            val = 0.0;
        }
        feature_vector.push_back(val);
    }
    std::vector<std::vector<double>> seq_data(sequence_length, feature_vector);
    torch::Tensor seq_tensor = torch::from_blob(seq_data.data(), {sequence_length, static_cast<int64_t>(FIXED_FEATURES.size())})
                                  .reshape({1, sequence_length, static_cast<int64_t>(FIXED_FEATURES.size())}).to(torch::kFloat32);
    if (!seq_tensor.isfinite().all().item<bool>()) {
        std::cerr << "Warning: Sequence tensor contains non-finite values" << std::endl;
    }

    // Create scalar features
    std::map<std::string, double> scalar_features = features;
    for (const auto& feature : LOW_IMPORTANCE_FEATURES) {
        scalar_features.erase(feature);
    }

    // Log transform skewed features
    for (const auto& feature : SKEWED_FEATURES) {
        if (scalar_features.find(feature) != scalar_features.end()) {
            double val = scalar_features[feature];
            double log_val = std::log1p(val);
            if (!std::isfinite(log_val)) {
                std::cerr << "Warning: Non-finite log value for " << feature << ": " << log_val << std::endl;
                log_val = 0.0;
            }
            scalar_features["log_" + feature] = log_val;
            scalar_features.erase(feature);
        }
    }

    // Create scalar vector
    std::vector<std::string> scalar_columns = scaler_params.feature_columns;
    std::vector<double> scalar_vector;
    for (const auto& col : scalar_columns) {
        double val = scalar_features[col];
        if (!std::isfinite(val)) {
            std::cerr << "Warning: Non-finite value for scalar feature " << col << ": " << val << std::endl;
            val = 0.0;
        }
        scalar_vector.push_back(val);
    }

    // Apply RobustScaler
    std::vector<double> scalar_scaled(scalar_vector.size());
    for (size_t i = 0; i < scalar_vector.size(); ++i) {
        if (scaler_params.x_scalar_scale[i] == 0) {
            std::cerr << "Error: Zero scale for feature " << scaler_params.feature_columns[i] << std::endl;
            scalar_scaled[i] = 0.0;
        } else {
            scalar_scaled[i] = (scalar_vector[i] - scaler_params.x_scalar_center[i]) / scaler_params.x_scalar_scale[i];
            if (!std::isfinite(scalar_scaled[i])) {
                std::cerr << "Warning: Non-finite scaled value for " << scaler_params.feature_columns[i]
                          << ": " << scalar_scaled[i]
                          << " (value=" << scalar_vector[i]
                          << ", center=" << scaler_params.x_scalar_center[i]
                          << ", scale=" << scaler_params.x_scalar_scale[i] << ")" << std::endl;
                scalar_scaled[i] = 0.0;
            }
        }
    }
    torch::Tensor scalar_tensor = torch::from_blob(scalar_scaled.data(), {1, static_cast<int64_t>(scalar_scaled.size())})
                                     .to(torch::kFloat32);
    if (!scalar_tensor.isfinite().all().item<bool>()) {
        std::cerr << "Warning: Scalar tensor contains non-finite values" << std::endl;
    }

    return {seq_tensor, scalar_tensor, features.at("execution_time_ms")};
}

int main(int argc, char* argv[]) {
    try {
        // Check for input JSON file
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <input_json_file>" << std::endl;
            return -1;
        }
        std::string input_json_file = argv[1];

        // Initialize CUDA if available
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "CUDA not available. Using CPU." << std::endl;
        }

        // Load scaler parameters
        ScalerParams scaler_params;
        try {
            scaler_params = load_scaler_params("scaler_params.json");
        } catch (const std::exception& e) {
            std::cerr << "Error loading scaler_params.json: " << e.what() << std::endl;
            return -1;
        }

        // Load the model
        torch::jit::script::Module module;
        try {
            module = torch::jit::load("model.pt");
            module.eval();
            module.to(device);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            return -1;
        }

        // Read JSON file
        std::ifstream ifs(input_json_file);
        if (!ifs.is_open()) {
            std::cerr << "Error: Could not open " << input_json_file << std::endl;
            return -1;
        }
        json json_data;
        try {
            ifs >> json_data;
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing " << input_json_file << ": " << e.what() << std::endl;
            return -1;
        }

        // Extract features
        auto features = extract_features(json_data);
        if (features["execution_time_ms"] <= 0 || !std::isfinite(features["execution_time_ms"])) {
            std::cerr << "Invalid execution time in " << input_json_file << std::endl;
            return -1;
        }

        // Preprocess features
        auto preprocessed = preprocess_features(features, scaler_params);
        auto seq_input = preprocessed.seq_input.to(device);
        auto scalar_input = preprocessed.scalar_input.to(device);

        // Print input tensor stats
        std::cout << "Sequence input min: " << seq_input.min().item<float>()
                  << ", max: " << seq_input.max().item<float>() << std::endl;
        std::cout << "Scalar input min: " << scalar_input.min().item<float>()
                  << ", max: " << scalar_input.max().item<float>() << std::endl;

        // Perform inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {seq_input, scalar_input};
        auto output = module.forward(inputs).toTensor();
        float pred_scaled = output.item<float>();
        std::cout << "Model output (scaled): " << pred_scaled << std::endl;

        // Inverse transform prediction
        float pred_transformed = (pred_scaled * scaler_params.y_scale[0]) + scaler_params.y_center[0];
        std::cout << "Inverse scaled output: " << pred_transformed << std::endl;
        double pred_actual = std::expm1(pred_transformed);
        std::cout << "Final prediction (expm1): " << pred_actual << std::endl;
        pred_actual = std::max(pred_actual, 0.0);

        // Output results
        std::cout << "File: " << input_json_file << "\n"
                  << "  Actual execution time: " << features["execution_time_ms"] << " ms\n"
                  << "  Predicted execution time: " << pred_actual << " ms\n"
                  << "  Error percentage: " << (features["execution_time_ms"] > 0 ?
                      std::abs(features["execution_time_ms"] - pred_actual) / features["execution_time_ms"] * 100 : 0.0) << "%\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
