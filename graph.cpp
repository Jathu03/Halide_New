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

// Struct to hold feature extraction results
struct FeatureResult {
    std::unordered_map<std::string, float> features;
    std::vector<float> node_vector;
    std::vector<float> scalar_vector;
};

// Feature extraction (adapted from Python extract_features for Graph_Output)
struct GraphFeatures {
    std::vector<std::string> node_feature_names;
    std::vector<std::string> scalar_feature_names;
    std::vector<std::string> skewed_features;
    std::vector<std::string> dropped_features;

    FeatureResult extract(const json& json_data) {
        std::cout << "Extracting features from JSON" << std::endl;
        FeatureResult result;
        auto& features = result.features;

        // Validate JSON structure
        if (!json_data.contains("without_extern") || !json_data["without_extern"].contains("global_features")) {
            throw std::runtime_error("JSON missing 'without_extern' or 'global_features'");
        }
        auto global_features = json_data["without_extern"]["global_features"];
        if (!global_features.contains("execution_time_ms")) {
            throw std::runtime_error("JSON missing 'execution_time_ms'");
        }
        features["execution_time_ms"] = global_features["execution_time_ms"].get<float>();
        features["cache_hits"] = global_features.contains("cache_hits") ? global_features["cache_hits"].get<float>() : 0.0f;
        features["cache_misses"] = global_features.contains("cache_misses") ? global_features["cache_misses"].get<float>() : 0.0f;

        // Extract node and edge counts
        auto nodes = json_data["without_extern"].contains("nodes") ? json_data["without_extern"]["nodes"] : json::array();
        auto edges = json_data["without_extern"].contains("edges") ? json_data["without_extern"]["edges"] : json::array();
        features["nodes_count"] = nodes.size();
        features["edges_count"] = edges.size();
        features["node_edge_ratio"] = features["nodes_count"] / (features["edges_count"] + 1e-8);

        // Extract operation counts and memory patterns
        std::unordered_map<std::string, float> op_counts;
        std::unordered_map<std::string, std::vector<float>> memory_patterns;
        for (const auto& pattern : {"transpose", "slice", "broadcast", "pointwise"}) {
            memory_patterns[pattern] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        for (const auto& node : nodes) {
            if (!node.contains("stages") || !node["stages"].is_array()) {
                continue;
            }
            for (const auto& stage : node["stages"]) {
                if (!stage.contains("pipeline_features")) {
                    continue;
                }
                auto pipeline = stage["pipeline_features"];
                // Operation histogram (Float type)
                if (pipeline.contains("op_histogram") && pipeline["op_histogram"].contains("Float")) {
                    for (const auto& [op, count] : pipeline["op_histogram"]["Float"].items()) {
                        std::string op_lower = op;
                        std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                        op_counts["op_" + op_lower] += count.get<float>();
                    }
                }
                // Memory access patterns (Float type)
                if (pipeline.contains("memory_access_patterns") && pipeline["memory_access_patterns"].contains("Float")) {
                    for (const auto& [pattern, values] : pipeline["memory_access_patterns"]["Float"].items()) {
                        std::string pattern_lower = pattern;
                        std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                        if (memory_patterns.count(pattern_lower) && values.is_array()) {
                            auto& vals = memory_patterns[pattern_lower];
                            for (size_t i = 0; i < std::min<size_t>(values.size(), 4); ++i) {
                                vals[i] += values[i].is_number() ? values[i].get<float>() : 0.0f;
                            }
                        }
                    }
                }
            }
        }
        features.insert(op_counts.begin(), op_counts.end());
        for (const auto& [pattern, values] : memory_patterns) {
            for (size_t i = 0; i < 4; ++i) {
                features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
            }
        }

        // Extract scheduling features
        std::vector<json> scheduling_features;
        for (const auto& node : nodes) {
            if (node.contains("stages")) {
                for (const auto& stage : node["stages"]) {
                    if (stage.contains("schedule_features")) {
                        scheduling_features.push_back(stage["schedule_features"]);
                    }
                }
            }
        }
        features["scheduling_count"] = scheduling_features.size();

        if (!scheduling_features.empty()) {
            std::vector<std::string> important_metrics = {
                "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
                "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
                "num_scalars", "num_vectors", "points_computed_total", "working_set"
            };
            for (const auto& metric : important_metrics) {
                float sum = 0.0f;
                for (const auto& sf : scheduling_features) {
                    sum += sf.contains(metric) ? sf[metric].get<float>() : 0.0f;
                }
                features["sched_" + metric] = sum;
            }
            features["total_bytes_at_production"] = features["sched_bytes_at_production"];
            features["total_vectors"] = features["sched_num_vectors"];
            float total_parallelism = 0.0f;
            for (const auto& sf : scheduling_features) {
                float inner = sf.contains("inner_parallelism") ? sf["inner_parallelism"].get<float>() : 0.0f;
                float outer = sf.contains("outer_parallelism") ? sf["outer_parallelism"].get<float>() : 1.0f;
                total_parallelism += inner * outer;
            }
            features["total_parallelism"] = total_parallelism;
            features["bytes_per_vector"] = features["total_vectors"] > 0 ? features["total_bytes_at_production"] / features["total_vectors"] : 0.0f;
            features["memory_pressure"] = features["sched_bytes_at_production"] > 0 ? features["sched_working_set"] / features["sched_bytes_at_production"] : 0.0f;
            features["bytes_per_parallelism"] = total_parallelism > 0 ? features["total_bytes_at_production"] / total_parallelism : 0.0f;
            features["nodes_per_schedule"] = features["scheduling_count"] > 0 ? features["nodes_count"] / features["scheduling_count"] : features["nodes_count"];
        }

        // Derived features
        features["op_diversity"] = std::count_if(op_counts.begin(), op_counts.end(), 
            [](const auto& p) { return p.second > 0; });
        features["computation_efficiency"] = features["execution_time_ms"] > 0 ? 
            features["sched_points_computed_total"] / features["execution_time_ms"] : 0.0f;
        features["bytes_processing_rate"] = features["execution_time_ms"] > 0 ? 
            features["total_bytes_at_production"] / features["execution_time_ms"] : 0.0f;
        features["memory_utilization_ratio"] = features["sched_bytes_at_production"] > 0 ? 
            features["sched_working_set"] / features["sched_bytes_at_production"] : 0.0f;

        // Log transform skewed features
        for (const auto& feature : skewed_features) {
            if (features.count(feature)) {
                features["log_" + feature] = std::log1p(features[feature]);
                features.erase(feature);
            }
        }

        // Create node and scalar feature vectors
        std::vector<float> node_vec;
        for (const auto& key : node_feature_names) {
            node_vec.push_back(features.count(key) ? features[key] : 0.0f);
        }
        std::vector<float> scalar_vec;
        for (const auto& key : scalar_feature_names) {
            if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                scalar_vec.push_back(features.count(key) ? features[key] : 0.0f);
            }
        }

        // Store vectors in result
        result.node_vector = node_vec;
        result.scalar_vector = scalar_vec;
        std::cout << "Extracted node features: " << node_vec.size() << ", scalar features: " << scalar_vec.size() << std::endl;
        return result;
    }
};

// Function to predict execution time from JSON file
float predict_execution_time(const std::string& json_file_path) {
    try {
        // Read JSON file
        std::ifstream input_file(json_file_path);
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open " + json_file_path);
        }
        json json_data;
        try {
            input_file >> json_data;
        } catch (const json::parse_error& e) {
            throw std::runtime_error("Failed to parse JSON from " + json_file_path + ": " + e.what());
        }
        std::cout << "Input JSON loaded from " << json_file_path << std::endl;

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

        if (scaler_node.center.size() != seq_input_size || scaler_scalar.center.size() != scalar_input_size) {
            throw std::runtime_error("Scaler dimensions do not match input sizes: node_center=" +
                                     std::to_string(scaler_node.center.size()) + ", scalar_center=" +
                                     std::to_string(scaler_scalar.center.size()));
        }

        // Extract features
        GraphFeatures extractor;
        extractor.node_feature_names = node_features;
        extractor.scalar_feature_names = scalar_features;
        extractor.skewed_features = skewed_features;
        extractor.dropped_features = dropped_features;

        auto feature_result = extractor.extract(json_data);
        auto node_vec = feature_result.node_vector;
        auto scalar_vec = feature_result.scalar_vector;

        // Scale features
        auto scaled_node = scaler_node.transform(node_vec);
        auto scaled_scalar = scaler_scalar.transform(scalar_vec);

        // Create sequence tensor (repeat node features for max_sequence_length)
        std::vector<float> seq_data(max_sequence_length * seq_input_size, 0.0f);
        for (int i = 0; i < max_sequence_length; ++i) {
            for (int j = 0; j < seq_input_size; ++j) {
                seq_data[i * seq_input_size + j] = scaled_node[j];
            }
        }
        torch::Tensor seq_tensor = torch::from_blob(
            seq_data.data(),
            {1, max_sequence_length, seq_input_size},
            torch::kFloat
        ).clone();

        torch::Tensor scalar_tensor = torch::from_blob(
            scaled_scalar.data(),
            {1, scalar_input_size},
            torch::kFloat
        ).clone();

        // Determine device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "CUDA is available, using GPU" << std::endl;
        } else {
            std::cout << "CUDA is not available, using CPU" << std::endl;
        }
        seq_tensor = seq_tensor.to(device);
        scalar_tensor = scalar_tensor.to(device);

        // Load model
        torch::jit::script::Module model;
        try {
            model = torch::jit::load("model.pt");
        } catch (const c10::Error& e) {
            throw std::runtime_error("Failed to load model.pt: " + std::string(e.what()));
        }
        model.eval();
        model.to(device);
        std::cout << "Model loaded and moved to " << (device.is_cuda() ? "CUDA" : "CPU") << " device" << std::endl;

        // Run inference
        std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
        auto output = model.forward(inputs).toTensor();
        float scaled_output = output.item<float>();
        std::cout << "Inference completed: scaled_output=" << scaled_output << std::endl;

        // Inverse transform output
        float log_output = scaler_y.inverse_transform(scaled_output, 0);
        float execution_time_ms = std::expm1(log_output);
        execution_time_ms = std::max(0.0f, execution_time_ms);
        std::cout << "Predicted execution time: " << execution_time_ms << " ms" << std::endl;

        return execution_time_ms;

    } catch (const c10::Error& e) {
        std::cerr << "PyTorch Error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

// Main function
int main(int argc, char* argv[]) {
    try {
        // Command-line argument for JSON file
        std::string json_file_path;
        if (argc > 1) {
            json_file_path = argv[1];
        } else {
            json_file_path = "converted_function_graph.json";
            std::cout << "No input file specified, using default: " << json_file_path << std::endl;
        }

        // Predict execution time
        float execution_time_ms = predict_execution_time(json_file_path);

        // Output result as JSON
        json output;
        output["execution_time_ms"] = execution_time_ms;
        std::cout << output.dump(4) << std::endl;

        return 0;

    } catch (const c10::Error& e) {
        std::cerr << "PyTorch Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
