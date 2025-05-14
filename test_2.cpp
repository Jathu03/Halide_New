#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>

using json = nlohmann::json;

// RobustScaler implementation
struct RobustScaler {
    std::vector<float> center;
    std::vector<float> scale;

    void load(const std::string& params_file) {
        std::cout << "Loading scaler from " << params_file << std::endl;
        std::ifstream file(params_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open " + params_file);
        }
        json params;
        file >> params;
        center = params["center"].get<std::vector<float>>();
        scale = params["scale"].get<std::vector<float>>();
        std::cout << "Scaler loaded: center size=" << center.size() << ", scale size=" << scale.size() << std::endl;
    }

    std::vector<float> transform(const std::vector<float>& input) {
        if (input.size() != center.size()) {
            throw std::runtime_error("Input size (" + std::to_string(input.size()) + ") does not match scaler size (" + std::to_string(center.size()) + ")");
        }
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = (input[i] - center[i]) / (scale[i] + 1e-8);
        }
        return output;
    }

    float inverse_transform(float input, size_t index) {
        if (index >= scale.size()) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for scaler");
        }
        return input * scale[index] + center[index];
    }
};

// Feature extraction
struct FeatureExtractor {
    std::vector<std::string> feature_names;
    std::vector<std::string> skewed_features;
    std::vector<std::string> dropped_features;

    std::unordered_map<std::string, float> extract(const json& json_data) {
        std::cout << "Extracting features from JSON" << std::endl;
        std::unordered_map<std::string, float> features;

        // Extract global features
        bool found_global = false;
        for (const auto& child : json_data["children"]) {
            if (child["name"] == "Global Features") {
                features["cache_hits"] = child.contains("cache_hits") ? child["cache_hits"].get<float>() : 0.0f;
                features["cache_misses"] = child.contains("cache_misses") ? child["cache_misses"].get<float>() : 0.0f;
                features["execution_time_ms"] = child.contains("execution_time_ms") ? child["execution_time_ms"].get<float>() : 0.0f;
                found_global = true;
                break;
            }
        }
        if (!found_global) {
            std::cout << "Warning: Global Features node not found" << std::endl;
        }

        // Extract op_histogram
        std::unordered_map<std::string, float> op_histogram;
        for (const auto& node : json_data["children"]) {
            if (node.contains("op_histogram")) {
                for (const auto& [op, count] : node["op_histogram"].items()) {
                    std::string op_lower = op;
                    std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                    op_histogram[op_lower] += count.get<float>();
                }
            }
        }
        for (const auto& [op, count] : op_histogram) {
            features["op_" + op] = count;
        }

        // Extract memory patterns
        std::unordered_map<std::string, std::vector<float>> memory_patterns;
        for (const auto& pattern : {"transpose", "slice", "broadcast", "pointwise"}) {
            memory_patterns[pattern] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        for (const auto& node : json_data["children"]) {
            if (node.contains("memory_patterns")) {
                for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                    std::string pattern_lower = pattern;
                    std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                    if (memory_patterns.count(pattern_lower)) {
                        std::vector<float> vals(4, 0.0f);
                        if (values.is_array() && values.size() == 4) {
                            for (size_t i = 0; i < 4; ++i) {
                                vals[i] = values[i].get<float>();
                            }
                        }
                        for (size_t i = 0; i < 4; ++i) {
                            memory_patterns[pattern_lower][i] += vals[i];
                        }
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
        std::unordered_map<std::string, float> scheduling_sums;
        float node_count = 0;
        for (const auto& node : json_data["children"]) {
            if (node.contains("scheduling")) {
                node_count += 1;
                for (const auto& key : scheduling_keys) {
                    scheduling_sums[key] += node["scheduling"].contains(key) ? node["scheduling"][key].get<float>() : 0.0f;
                }
            }
        }
        for (const auto& key : scheduling_keys) {
            if (key == "inner_parallelism" || key == "outer_parallelism") {
                features["sched_" + key] = node_count > 0 ? scheduling_sums[key] / node_count : 0.0f;
            } else {
                features["sched_" + key] = scheduling_sums[key];
            }
        }

        // Derived features
        features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"];
        features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"];
        features["total_bytes_at_production"] = features["sched_bytes_at_production"];
        features["total_vectors"] = features["sched_num_vectors"];
        features["computation_efficiency"] = features["sched_bytes_at_realization"] > 0 ?
            features["sched_points_computed_total"] / features["sched_bytes_at_realization"] : 0.0f;
        features["memory_pressure"] = features["sched_bytes_at_root"] > 0 ?
            features["sched_working_set"] / features["sched_bytes_at_root"] : 0.0f;
        features["memory_utilization_ratio"] = features["sched_bytes_at_task"] > 0 ?
            features["sched_unique_bytes_read_per_realization"] / features["sched_bytes_at_task"] : 0.0f;
        features["bytes_processing_rate"] = features["execution_time_ms"] > 0 ?
            features["sched_bytes_at_realization"] / features["execution_time_ms"] : 0.0f;
        features["bytes_per_parallelism"] = features["total_parallelism"] > 0 ?
            features["sched_bytes_at_task"] / features["total_parallelism"] : 0.0f;
        features["bytes_per_vector"] = features["sched_num_vectors"] > 0 ?
            features["sched_bytes_at_realization"] / features["sched_num_vectors"] : 0.0f;
        float nodes_count = json_data["children"].size();
        float edges_count = 0;
        for (const auto& node : json_data["children"]) {
            edges_count += node.contains("children") ? node["children"].size() : 0;
        }
        features["nodes_count"] = nodes_count;
        features["edges_count"] = edges_count;
        features["node_edge_ratio"] = edges_count > 0 ? nodes_count / (edges_count + 1) : nodes_count;
        features["nodes_per_schedule"] = features["scheduling_count"] > 0 ?
            nodes_count / features["scheduling_count"] : nodes_count;
        std::set<std::string> ops;
        for (const auto& [key, value] : features) {
            if (key.find("op_") == 0 && value > 0) {
                ops.insert(key);
            }
        }
        features["op_diversity"] = ops.size();

        // Apply log transformation for skewed features
        for (const auto& feature : skewed_features) {
            if (features.count(feature)) {
                features["log_" + feature] = std::log1p(features[feature]);
                features.erase(feature);
            }
        }

        // Create fixed-length feature vector
        std::unordered_map<std::string, float> ordered_features;
        for (const auto& key : feature_names) {
            ordered_features[key] = features.count(key) ? features[key] : 0.0f;
        }

        return ordered_features;
    }
};

// Main function
int main(int argc, char* argv[]) {
    try {
        // Parse input file path
        std::string input_file_path = "tree_representation.json";
        if (argc > 1) {
            input_file_path = argv[1];
        }
        std::cout << "Input file: " << input_file_path << std::endl;

        // Load metadata
        std::ifstream metadata_file("model_metadata.json");
        if (!metadata_file.is_open()) {
            throw std::runtime_error("Failed to open model_metadata.json");
        }
        json metadata;
        metadata_file >> metadata;
        int max_sequence_length = metadata["max_sequence_length"].get<int>();
        int seq_input_size = metadata["seq_input_size"].get<int>();
        int scalar_input_size = metadata["scalar_input_size"].get<int>();
        std::vector<std::string> node_features = metadata["node_features"].get<std::vector<std::string>>();
        std::vector<std::string> scalar_features = metadata["scalar_features"].get<std::vector<std::string>>();
        std::vector<std::string> skewed_features = metadata["skewed_features"].get<std::vector<std::string>>();
        std::vector<std::string> dropped_features = metadata["dropped_features"].get<std::vector<std::string>>();
        std::cout << "Metadata loaded: seq_input_size=" << seq_input_size << ", scalar_input_size=" << scalar_input_size << std::endl;

        // Load scalers
        RobustScaler scaler_node, scaler_scalar, scaler_y;
        scaler_node.load("scaler_node_params.json");
        scaler_scalar.load("scaler_scalar_params.json");
        scaler_y.load("scaler_y_params.json");
        if (scaler_node.center.size() != seq_input_size) {
            throw std::runtime_error("Node scaler dimension (" + std::to_string(scaler_node.center.size()) + ") does not match seq_input_size (" + std::to_string(seq_input_size) + ")");
        }
        if (scaler_scalar.center.size() != scalar_input_size) {
            throw std::runtime_error("Scalar scaler dimension (" + std::to_string(scaler_scalar.center.size()) + ") does not match scalar_input_size (" + std::to_string(scalar_input_size) + ")");
        }

        // Load JSON input
        std::ifstream input_file(input_file_path);
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open " + input_file_path);
        }
        json json_data;
        input_file >> json_data;
        std::cout << "Input JSON loaded" << std::endl;

        // Extract features
        FeatureExtractor extractor;
        extractor.feature_names = node_features;
        extractor.skewed_features = skewed_features;
        extractor.dropped_features = dropped_features;
        auto features = extractor.extract(json_data);

        // Create sequence input
        std::vector<float> feature_vec;
        for (const auto& key : node_features) {
            feature_vec.push_back(features[key]);
        }
        auto scaled_features = scaler_node.transform(feature_vec);
        std::vector<float> seq_data(max_sequence_length * seq_input_size, 0.0f);
        for (int i = 0; i < max_sequence_length; ++i) {
            for (int j = 0; j < seq_input_size; ++j) {
                seq_data[i * seq_input_size + j] = scaled_features[j];
            }
        }

        // Create scalar input
        std::vector<float> scalar_vec;
        for (const auto& key : scalar_features) {
            if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                scalar_vec.push_back(features[key]);
            }
        }
        auto scaled_scalar = scaler_scalar.transform(scalar_vec);

        // Determine device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }

        // Load model
        torch::jit::script::Module model;
        try {
            model = torch::jit::load("model.pt", device);
            model.eval();
            model.to(device);
            std::cout << "Model loaded and moved to device" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model.pt: " + std::string(e.what()));
        }

        // Create tensors
        torch::Tensor seq_tensor = torch::from_blob(
            seq_data.data(),
            {1, max_sequence_length, seq_input_size},
            torch::kFloat
        ).clone().to(device);
        torch::Tensor scalar_tensor = torch::from_blob(
            scaled_scalar.data(),
            {1, scalar_input_size},
            torch::kFloat
        ).clone().to(device);

        // Run inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
        auto output = model.forward(inputs).toTensor();
        float scaled_output = output.item<float>();

        // Inverse transform output
        float log_output = scaler_y.inverse_transform(scaled_output, 0);
        float execution_time_ms = std::expm1(log_output);
        execution_time_ms = std::max(0.0f, execution_time_ms);

        std::cout << "Predicted execution time: " << execution_time_ms << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
