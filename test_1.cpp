#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>

using json = nlohmann::json;

// RobustScaler implementation
struct RobustScaler {
    std::vector<float> center;
    std::vector<float> scale;

    void load(const std::string& params_file) {
        std::ifstream file(params_file);
        json params;
        file >> params;
        center = params["center"].get<std::vector<float>>();
        scale = params["scale"].get<std::vector<float>>();
    }

    std::vector<float> transform(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = (input[i] - center[i]) / scale[i];
        }
        return output;
    }

    float inverse_transform(float input, size_t index) {
        return input * scale[index] + center[index];
    }
};

// Node feature extraction (simplified, adapt to your JSON structure)
struct NodeFeatures {
    std::vector<std::string> feature_names;
    std::unordered_map<std::string, float> extract_node(const json& node) {
        std::unordered_map<std::string, float> features;
        // Cache features
        features["cache_hits"] = node.contains("cache_hits") ? node["cache_hits"].get<float>() : 0.0f;
        features["cache_misses"] = node.contains("cache_misses") ? node["cache_misses"].get<float>() : 0.0f;

        // Scheduling features
        if (node.contains("scheduling")) {
            auto sched = node["scheduling"];
            for (const auto& key : {
                "num_realizations", "num_productions", "points_computed_total", "innermost_loop_extent",
                "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
                "bytes_at_root", "unique_bytes_read_per_realization", "working_set", "vector_size",
                "num_vectors", "num_scalars", "bytes_at_task", "working_set_at_task",
                "working_set_at_production", "working_set_at_realization", "working_set_at_root"
            }) {
                features["sched_" + std::string(key)] = sched.contains(key) ? sched[key].get<float>() : 0.0f;
            }
        }

        // Op histogram
        if (node.contains("op_histogram")) {
            for (const auto& [op, count] : node["op_histogram"].items()) {
                features["op_" + op] = count.get<float>();
            }
        }

        // Memory patterns
        if (node.contains("memory_patterns")) {
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                for (size_t i = 0; i < 4; ++i) {
                    features["memory_" + pattern + "_" + std::to_string(i)] = values[i].get<float>();
                }
            }
        }

        // Create fixed-length feature vector
        std::unordered_map<std::string, float> ordered_features;
        for (const auto& key : feature_names) {
            ordered_features[key] = features[key];
        }
        return ordered_features;
    }
};

// Scalar feature extraction (simplified, adapt to your needs)
struct ScalarFeatures {
    std::vector<std::string> feature_names;
    std::vector<std::string> skewed_features;
    std::unordered_map<std::string, float> extract(const json& json_data, const std::vector<json>& nodes) {
        std::unordered_map<std::string, float> features;
        // Global features
        for (const auto& child : json_data["children"]) {
            if (child["name"] == "Global Features") {
                features["execution_time_ms"] = child["execution_time_ms"].get<float>();
                break;
            }
        }

        // Derived features
        float node_count = nodes.size();
        float scheduling_count = 0, total_parallelism = 0, total_bytes_at_production = 0, total_vectors = 0;
        float points_computed_total = 0, bytes_at_realization = 0, working_set = 0, bytes_at_root = 0;
        float unique_bytes_read_per_realization = 0, bytes_at_task = 0;

        for (const auto& node : nodes) {
            if (node.contains("scheduling")) {
                auto sched = node["scheduling"];
                scheduling_count += (sched["num_realizations"].get<float>() + sched["num_productions"].get<float>());
                total_parallelism += (sched["inner_parallelism"].get<float>() + sched["outer_parallelism"].get<float>());
                total_bytes_at_production += sched["bytes_at_production"].get<float>();
                total_vectors += sched["num_vectors"].get<float>();
                points_computed_total += sched["points_computed_total"].get<float>();
                bytes_at_realization += sched["bytes_at_realization"].get<float>();
                working_set += sched["working_set"].get<float>();
                bytes_at_root += sched["bytes_at_root"].get<float>();
                unique_bytes_read_per_realization += sched["unique_bytes_read_per_realization"].get<float>();
                bytes_at_task += sched["bytes_at_task"].get<float>();
            }
        }

        features["total_parallelism"] = total_parallelism / std::max(node_count, 1.0f);
        features["scheduling_count"] = scheduling_count;
        features["total_bytes_at_production"] = total_bytes_at_production;
        features["total_vectors"] = total_vectors;
        features["computation_efficiency"] = points_computed_total / std::max(bytes_at_realization, 1.0f);
        features["memory_pressure"] = working_set / std::max(bytes_at_root, 1.0f);
        features["memory_utilization_ratio"] = unique_bytes_read_per_realization / std::max(bytes_at_task, 1.0f);
        features["bytes_processing_rate"] = bytes_at_realization / std::max(features["execution_time_ms"], 1.0f);
        features["bytes_per_parallelism"] = bytes_at_task / std::max(features["total_parallelism"], 1.0f);
        features["bytes_per_vector"] = bytes_at_realization / std::max(features["total_vectors"], 1.0f);
        features["nodes_count"] = node_count;
        float edges_count = 0;
        for (const auto& node : nodes) {
            edges_count += node["children"].size();
        }
        features["edges_count"] = edges_count;
        features["node_edge_ratio"] = node_count / std::max(edges_count + 1, 1.0f);
        features["nodes_per_schedule"] = node_count / std::max(scheduling_count, 1.0f);

        // Op diversity
        std::unordered_set<std::string> ops;
        for (const auto& node : nodes) {
            if (node.contains("op_histogram")) {
                for (const auto& [op, count] : node["op_histogram"].items()) {
                    if (count.get<float>() > 0) {
                        ops.insert(op);
                    }
                }
            }
        }
        features["op_diversity"] = ops.size();

        // Apply log transformation for skewed features
        for (const auto& feature : skewed_features) {
            if (features.contains(feature)) {
                features["log_" + feature] = std::log1p(features[feature]);
                features.erase(feature);
            }
        }

        // Create fixed-length feature vector
        std::unordered_map<std::string, float> ordered_features;
        for (const auto& key : feature_names) {
            ordered_features[key] = features[key];
        }
        return ordered_features;
    }
};

int main() {
    // Load metadata
    std::ifstream metadata_file("model_metadata.json");
    json metadata;
    metadata_file >> metadata;
    int max_sequence_length = metadata["max_sequence_length"].get<int>();
    int seq_input_size = metadata["seq_input_size"].get<int>();
    int scalar_input_size = metadata["scalar_input_size"].get<int>();
    std::vector<std::string> node_features = metadata["node_features"].get<std::vector<std::string>>();
    std::vector<std::string> scalar_features = metadata["scalar_features"].get<std::vector<std::string>>();
    std::vector<std::string> skewed_features = metadata["skewed_features"].get<std::vector<std::string>>();

    // Load scalers
    RobustScaler scaler_node, scaler_scalar, scaler_y;
    scaler_node.load("scaler_node_params.json");
    scaler_scalar.load("scaler_scalar_params.json");
    scaler_y.load("scaler_y_params.json");

    // Load model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("recursive_model.pt");
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    // Load input JSON (example: tree_representation.json)
    std::ifstream input_file("tree_representation.json");
    json json_data;
    input_file >> input_file;

    // Extract features
    NodeFeatures node_extractor;
    node_extractor.feature_names = node_features;
    ScalarFeatures scalar_extractor;
    scalar_extractor.feature_names = scalar_features;
    scalar_extractor.skewed_features = skewed_features;

    std::vector<std::vector<float>> node_sequences;
    auto traverse_nodes = [&](const json& node, auto&& traverse_nodes) -> void {
        auto features = node_extractor.extract_node(node);
        std::vector<float> feature_vec;
        for (const auto& key : node_features) {
            feature_vec.push_back(features[key]);
        }
        node_sequences.push_back(feature_vec);
        for (const auto& child : node["children"]) {
            traverse_nodes(child, traverse_nodes);
        }
    };
    traverse_nodes(json_data, traverse_nodes);

    // Scale node features
    std::vector<std::vector<float>> scaled_node_sequences;
    for (const auto& node : node_sequences) {
        scaled_node_sequences.push_back(scaler_node.transform(node));
    }

    // Pad sequences
    std::vector<std::vector<float>> padded_sequences;
    for (const auto& seq : scaled_node_sequences) {
        std::vector<float> padded_seq(seq.begin(), seq.end());
        padded_seq.resize(max_sequence_length * seq_input_size, 0.0f);
        padded_sequences.push_back(padded_seq);
    }

    // Convert to tensor
    torch::Tensor seq_tensor = torch::from_blob(
        padded_sequences[0].data(),
        {1, max_sequence_length, seq_input_size},
        torch::kFloat
    );

    // Extract and scale scalar features
    std::vector<json> nodes;
    for (const auto& child : json_data["children"]) {
        if (child["name"] != "Global Features") {
            nodes.push_back(child);
        }
    }
    auto scalar_features = scalar_extractor.extract(json_data, nodes);
    std::vector<float> scalar_vec;
    for (const auto& key : scalar_features) {
        scalar_vec.push_back(scalar_features[key]);
    }
    auto scaled_scalar = scaler_scalar.transform(scalar_vec);
    torch::Tensor scalar_tensor = torch::from_blob(
        scaled_scalar.data(),
        {1, scalar_input_size},
        torch::kFloat
    );

    // Perform inference
    std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
    auto output = model.forward(inputs).toTensor();

    // Postprocess output
    float scaled_output = output.item<float>();
    float log_output = scaler_y.inverse_transform(scaled_output, 0);
    float execution_time_ms = std::expm1(log_output);

    std::cout << "Predicted execution time: " << execution_time_ms << " ms" << std::endl;

    return 0;
}
