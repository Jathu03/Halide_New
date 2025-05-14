#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace inference {

struct RobustScaler {
    std::vector<float> center;
    std::vector<float> scale;

    void load(const std::string& params_file) {
        std::ifstream file(params_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open scaler file: " + params_file);
        }
        json params;
        file >> params;
        center = params["center"].get<std::vector<float>>();
        scale = params["scale"].get<std::vector<float>>();
    }

    std::vector<float> transform(const std::vector<float>& input) const {
        if (input.size() != center.size()) {
            throw std::runtime_error("Input size (" + std::to_string(input.size()) + 
                                    ") does not match scaler size (" + std::to_string(center.size()) + ")");
        }
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = (input[i] - center[i]) / (scale[i] + 1e-8);
        }
        return output;
    }

    float inverse_transform(float input, size_t index) const {
        if (index >= scale.size()) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for scaler");
        }
        return input * scale[index] + center[index];
    }
};

struct NodeFeatures {
    std::vector<std::string> feature_names;

    std::unordered_map<std::string, float> extract(const json& node) const {
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

        // Operation histogram
        std::unordered_map<std::string, float> op_histogram;
        if (node.contains("op_histogram")) {
            for (const auto& [op, count] : node["op_histogram"].items()) {
                try {
                    op_histogram[op] = count.get<float>();
                } catch (const std::exception&) {
                    // Ignore invalid op_histogram values
                }
            }
        }
        for (const auto& op : {
            "add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "or", "and", "not",
            "min", "max", "constant", "variable", "funccall", "imagecall", "externcall", "let", "param"
        }) {
            features["op_" + std::string(op)] = op_histogram.count(op) ? op_histogram[op] : 0.0f;
        }

        // Memory patterns
        std::unordered_map<std::string, std::vector<float>> memory_patterns;
        for (const auto& pattern : {"pointwise", "transpose", "broadcast", "slice"}) {
            memory_patterns[pattern] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        if (node.contains("memory_patterns") && node["memory_patterns"].is_object()) {
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                std::string pattern_lower = pattern;
                std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                if (memory_patterns.count(pattern_lower) && values.is_array()) {
                    std::vector<float> vals(4, 0.0f);
                    for (size_t i = 0; i < std::min<size_t>(values.size(), 4); ++i) {
                        if (values[i].is_number()) {
                            vals[i] = values[i].get<float>();
                        }
                    }
                    memory_patterns[pattern_lower] = vals;
                }
            }
        }
        for (const auto& pattern : {"pointwise", "transpose", "broadcast", "slice"}) {
            auto values = memory_patterns[pattern];
            for (size_t i = 0; i < 4; ++i) {
                features["memory_" + std::string(pattern) + "_" + std::to_string(i)] = values[i];
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

struct ScalarFeatures {
    std::vector<std::string> feature_names;
    std::vector<std::string> skewed_features;
    std::vector<std::string> dropped_features;

    std::unordered_map<std::string, float> extract(const json& json_data) const {
        std::unordered_map<std::string, float> features;

        bool found_global = false;
        for (const auto& child : json_data["children"]) {
            if (child["name"] == "Global Features") {
                features["execution_time_ms"] = child["execution_time_ms"].get<float>();
                found_global = true;
                break;
            }
        }
        if (!found_global) {
            features["execution_time_ms"] = 0.0f;
        }

        std::vector<json> nodes;
        for (const auto& child : json_data["children"]) {
            if (child["name"] != "Global Features") {
                nodes.push_back(child);
            }
        }

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

        features["total_parallelism"] = node_count > 0 ? total_parallelism / node_count : 0.0f;
        features["scheduling_count"] = scheduling_count;
        features["total_bytes_at_production"] = total_bytes_at_production;
        features["total_vectors"] = total_vectors;
        features["computation_efficiency"] = bytes_at_realization > 0 ? points_computed_total / bytes_at_realization : 0.0f;
        features["memory_pressure"] = bytes_at_root > 0 ? working_set / bytes_at_root : 0.0f;
        features["memory_utilization_ratio"] = bytes_at_task > 0 ? unique_bytes_read_per_realization / bytes_at_task : 0.0f;
        features["bytes_processing_rate"] = features["execution_time_ms"] > 0 ? bytes_at_realization / features["execution_time_ms"] : 0.0f;
        features["bytes_per_parallelism"] = total_parallelism > 0 ? bytes_at_task / total_parallelism : 0.0f;
        features["bytes_per_vector"] = total_vectors > 0 ? bytes_at_realization / total_vectors : 0.0f;
        features["nodes_count"] = node_count;
        float edges_count = 0;
        for (const auto& node : nodes) {
            edges_count += node["children"].size();
        }
        features["edges_count"] = edges_count;
        features["node_edge_ratio"] = edges_count > 0 ? node_count / (edges_count + 1) : node_count;
        features["nodes_per_schedule"] = scheduling_count > 0 ? node_count / scheduling_count : node_count;

        std::set<std::string> ops;
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

        for (const auto& feature : skewed_features) {
            if (features.count(feature)) {
                features["log_" + feature] = std::log1p(features[feature]);
                features.erase(feature);
            }
        }

        std::unordered_map<std::string, float> ordered_features;
        for (const auto& key : feature_names) {
            if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                ordered_features[key] = features.count(key) ? features[key] : 0.0f;
            }
        }
        return ordered_features;
    }
};

class Predictor {
private:
    torch::jit::script::Module model;
    RobustScaler scaler_node, scaler_scalar, scaler_y;
    NodeFeatures node_extractor;
    ScalarFeatures scalar_extractor;
    torch::Device device;
    int max_sequence_length;
    int seq_input_size;
    int scalar_input_size;

    void load_metadata(const std::string& metadata_file) {
        std::ifstream file(metadata_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open metadata file: " + metadata_file);
        }
        json metadata;
        file >> metadata;

        max_sequence_length = metadata["max_sequence_length"].get<int>();
        seq_input_size = metadata["seq_input_size"].get<int>();
        scalar_input_size = metadata["scalar_input_size"].get<int>();
        node_extractor.feature_names = metadata["node_features"].get<std::vector<std::string>>();
        scalar_extractor.feature_names = metadata["scalar_features"].get<std::vector<std::string>>();
        scalar_extractor.skewed_features = metadata["skewed_features"].get<std::vector<std::string>>();
        scalar_extractor.dropped_features = metadata["dropped_features"].get<std::vector<std::string>>();
    }

public:
    Predictor(const std::string& model_path, const std::string& metadata_path,
              const std::string& scaler_node_path, const std::string& scaler_scalar_path,
              const std::string& scaler_y_path) : device(torch::kCPU) {
        // Check CUDA availability
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
        }

        // Load model
        model = torch::jit::load(model_path);
        model.eval();
        model.to(device);

        // Load scalers and metadata
        scaler_node.load(scaler_node_path);
        scaler_scalar.load(scaler_scalar_path);
        scaler_y.load(scaler_y_path);
        load_metadata(metadata_path);

        // Validate scaler dimensions
        if (scaler_node.center.size() != seq_input_size || scaler_scalar.center.size() != scalar_input_size) {
            throw std::runtime_error("Scaler dimensions do not match input sizes: node_center=" +
                                     std::to_string(scaler_node.center.size()) + ", scalar_center=" +
                                     std::to_string(scaler_scalar.center.size()));
        }
    }

    float predict_execution_time(const std::string& json_str) {
        // Parse JSON string
        json json_data;
        try {
            json_data = json::parse(json_str);
        } catch (const json::parse_error& e) {
            throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
        }

        // Extract node features
        std::vector<std::vector<float>> node_sequences;
        auto traverse_nodes = [&](const json& node, auto&& traverse_nodes) -> void {
            auto features = node_extractor.extract(node);
            std::vector<float> feature_vec;
            for (const auto& key : node_extractor.feature_names) {
                feature_vec.push_back(features[key]);
            }
            node_sequences.push_back(feature_vec);
            if (node.contains("children") && node["children"].is_array()) {
                for (const auto& child : node["children"]) {
                    traverse_nodes(child, traverse_nodes);
                }
            }
        };
        traverse_nodes(json_data, traverse_nodes);

        if (node_sequences.empty()) {
            throw std::runtime_error("No nodes extracted from JSON");
        }

        // Scale node features
        std::vector<std::vector<float>> scaled_node_sequences;
        for (const auto& node : node_sequences) {
            scaled_node_sequences.push_back(scaler_node.transform(node));
        }

        // Create sequence tensor
        std::vector<float> padded_data(max_sequence_length * seq_input_size, 0.0f);
        size_t nodes_to_copy = std::min(scaled_node_sequences.size(), static_cast<size_t>(max_sequence_length));
        for (size_t i = 0; i < nodes_to_copy; ++i) {
            for (size_t j = 0; j < seq_input_size; ++j) {
                padded_data[i * seq_input_size + j] = scaled_node_sequences[i][j];
            }
        }

        torch::Tensor seq_tensor = torch::from_blob(
            padded_data.data(),
            {1, max_sequence_length, seq_input_size},
            torch::kFloat
        ).clone().to(device);

        // Extract and scale scalar features
        auto scalar_features_map = scalar_extractor.extract(json_data);
        std::vector<float> scalar_vec;
        for (const auto& key : scalar_extractor.feature_names) {
            if (std::find(scalar_extractor.dropped_features.begin(), 
                          scalar_extractor.dropped_features.end(), key) == scalar_extractor.dropped_features.end()) {
                scalar_vec.push_back(scalar_features_map[key]);
            }
        }
        auto scaled_scalar = scaler_scalar.transform(scalar_vec);

        torch::Tensor scalar_tensor = torch::from_blob(
            scaled_scalar.data(),
            {1, scalar_input_size},
            torch::kFloat
        ).clone().to(device);

        // Run inference
        std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
        auto output = model.forward(inputs).toTensor();
        float scaled_output = output.item<float>();

        // Inverse transform output
        float log_output = scaler_y.inverse_transform(scaled_output, 0);
        float execution_time_ms = std::expm1(log_output);
        return std::max(0.0f, execution_time_ms);
    }
};

} // namespace inference
