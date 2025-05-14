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
#include <chrono>
#include <numeric>

using json = nlohmann::json;

// RobustScaler implementation
struct RobustScaler {
    std::vector<double> center;
    std::vector<double> scale;

    void load(const std::string& params_file) {
        std::cout << "Loading scaler from " << params_file << std::endl;
        std::ifstream file(params_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open " + params_file);
        }
        json params;
        file >> params;
        
        // Convert to double for better precision
        auto center_float = params["center"].get<std::vector<float>>();
        auto scale_float = params["scale"].get<std::vector<float>>();
        
        center.resize(center_float.size());
        scale.resize(scale_float.size());
        
        for (size_t i = 0; i < center_float.size(); ++i) {
            center[i] = static_cast<double>(center_float[i]);
            scale[i] = static_cast<double>(scale_float[i]);
        }
        
        std::cout << "Scaler loaded: center size=" << center.size() << ", scale size=" << scale.size() << std::endl;
    }

    std::vector<double> transform(const std::vector<double>& input) {
        if (input.size() != center.size()) {
            throw std::runtime_error("Input size (" + std::to_string(input.size()) + ") does not match scaler size (" + std::to_string(center.size()) + ")");
        }
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = (input[i] - center[i]) / (scale[i] + 1e-8);
        }
        return output;
    }

    double inverse_transform(double input, size_t index) {
        if (index >= scale.size()) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for scaler");
        }
        return input * scale[index] + center[index];
    }
};

// Enhanced Node feature extraction with additional features
struct NodeFeatures {
    std::vector<std::string> feature_names;

    // Calculate memory access pattern score based on memory patterns
    double calculateMemoryAccessScore(const json& node) {
        double score = 0.0;
        if (node.contains("memory_patterns") && node["memory_patterns"].is_object()) {
            auto& patterns = node["memory_patterns"];
            
            // Prioritize patterns based on efficiency (pointwise is most efficient)
            if (patterns.contains("pointwise") && patterns["pointwise"].is_array()) {
                auto& pw = patterns["pointwise"];
                score += pw.size() > 0 ? pw[0].get<double>() * 1.0 : 0.0;
            }
            
            if (patterns.contains("broadcast") && patterns["broadcast"].is_array()) {
                auto& bc = patterns["broadcast"];
                score += bc.size() > 0 ? bc[0].get<double>() * 0.8 : 0.0;
            }
            
            if (patterns.contains("slice") && patterns["slice"].is_array()) {
                auto& sl = patterns["slice"];
                score += sl.size() > 0 ? sl[0].get<double>() * 0.6 : 0.0;
            }
            
            if (patterns.contains("transpose") && patterns["transpose"].is_array()) {
                auto& tr = patterns["transpose"];
                score += tr.size() > 0 ? tr[0].get<double>() * 0.4 : 0.0;
            }
        }
        return score;
    }

    // Calculate computation intensity (ratio of compute ops to memory ops)
    double calculateComputationIntensity(const json& node) {
        double compute_ops = 0.0;
        double memory_ops = 0.0;
        
        if (node.contains("op_histogram")) {
            auto& ops = node["op_histogram"];
            
            // Compute operations
            for (const auto& op : {"add", "sub", "mul", "div", "mod", "min", "max"}) {
                if (ops.contains(op)) {
                    compute_ops += ops[op].get<double>();
                }
            }
            
            // Memory operations
            for (const auto& op : {"variable", "imagecall", "externcall", "let", "param"}) {
                if (ops.contains(op)) {
                    memory_ops += ops[op].get<double>();
                }
            }
        }
        
        return memory_ops > 0 ? compute_ops / memory_ops : compute_ops;
    }

    // Calculate data locality score
    double calculateDataLocalityScore(const json& node) {
        double score = 0.0;
        if (node.contains("scheduling")) {
            auto& sched = node["scheduling"];
            
            // Higher working set at task level indicates better locality
            if (sched.contains("working_set_at_task") && sched.contains("working_set")) {
                double ws_task = sched["working_set_at_task"].get<double>();
                double ws_total = sched["working_set"].get<double>();
                
                if (ws_total > 0) {
                    score = ws_task / ws_total;
                }
            }
        }
        return score;
    }

    std::unordered_map<std::string, double> extract(const json& node) {
        std::cout << "Extracting features for node: " << (node.contains("name") ? node["name"].get<std::string>() : "unnamed") << std::endl;
        std::unordered_map<std::string, double> features;

        // Cache features
        features["cache_hits"] = node.contains("cache_hits") ? node["cache_hits"].get<double>() : 0.0;
        features["cache_misses"] = node.contains("cache_misses") ? node["cache_misses"].get<double>() : 0.0;
        features["cache_ratio"] = features["cache_hits"] + features["cache_misses"] > 0 ? 
                                  features["cache_hits"] / (features["cache_hits"] + features["cache_misses"]) : 0.0;

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
                features["sched_" + std::string(key)] = sched.contains(key) ? sched[key].get<double>() : 0.0;
            }
            
            // Add derived scheduling features
            if (sched.contains("inner_parallelism") && sched.contains("outer_parallelism")) {
                features["sched_total_parallelism"] = sched["inner_parallelism"].get<double>() + 
                                                     sched["outer_parallelism"].get<double>();
            }
            
            if (sched.contains("bytes_at_realization") && sched.contains("points_computed_total")) {
                features["sched_bytes_per_point"] = sched["points_computed_total"].get<double>() > 0 ?
                                                  sched["bytes_at_realization"].get<double>() / 
                                                  sched["points_computed_total"].get<double>() : 0.0;
            }
        } else {
            std::cout << "Warning: Node missing scheduling field" << std::endl;
        }

        // Operation histogram
        std::unordered_map<std::string, double> op_histogram;
        if (node.contains("op_histogram")) {
            for (const auto& [op, count] : node["op_histogram"].items()) {
                try {
                    op_histogram[op] = count.get<double>();
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid op_histogram value for " << op << ": " << e.what() << std::endl;
                }
            }
        }
        
        double total_ops = 0.0;
        for (const auto& op : {
            "add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "or", "and", "not",
            "min", "max", "constant", "variable", "funccall", "imagecall", "externcall", "let", "param"
        }) {
            features["op_" + std::string(op)] = op_histogram.count(op) ? op_histogram[op] : 0.0;
            total_ops += features["op_" + std::string(op)];
        }
        
        // Add operation ratios for better feature representation
        if (total_ops > 0) {
            for (const auto& op : {
                "add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "or", "and", "not",
                "min", "max", "constant", "variable", "funccall", "imagecall", "externcall", "let", "param"
            }) {
                features["op_ratio_" + std::string(op)] = features["op_" + std::string(op)] / total_ops;
            }
        }

        // Memory patterns
        std::unordered_map<std::string, std::vector<double>> memory_patterns;
        for (const auto& pattern : {"pointwise", "transpose", "broadcast", "slice"}) {
            memory_patterns[pattern] = {0.0, 0.0, 0.0, 0.0};
        }
        
        if (node.contains("memory_patterns") && node["memory_patterns"].is_object()) {
            std::cout << "Processing memory_patterns" << std::endl;
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                std::string pattern_lower = pattern;
                std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                
                if (memory_patterns.count(pattern_lower)) {
                    std::vector<double> vals(4, 0.0);
                    
                    // Safely check if values is an array and has elements
                    if (values.is_array()) {
                        size_t array_size = values.size();
                        std::cout << "memory_patterns[" << pattern_lower << "] has " << array_size << " elements" << std::endl;
                        
                        for (size_t i = 0; i < std::min<size_t>(array_size, 4); ++i) {
                            try {
                                if (i < array_size && values[i].is_number()) {
                                    vals[i] = values[i].get<double>();
                                } else {
                                    std::cout << "Warning: Non-numeric or missing value in memory_patterns[" << pattern_lower << "][" << i << "]" << std::endl;
                                }
                            } catch (const std::exception& e) {
                                std::cout << "Warning: Invalid value in memory_patterns[" << pattern_lower << "][" << i << "]: " << e.what() << std::endl;
                            }
                        }
                    } else {
                        std::cout << "Warning: memory_patterns[" << pattern_lower << "] is not an array" << std::endl;
                    }
                    
                    memory_patterns[pattern_lower] = vals;
                } else {
                    std::cout << "Warning: Unknown memory pattern key: " << pattern_lower << std::endl;
                }
            }
        } else {
            std::cout << "Warning: Node missing memory_patterns field or not an object" << std::endl;
        }
        
        for (const auto& pattern : {"pointwise", "transpose", "broadcast", "slice"}) {
            auto values = memory_patterns[pattern];
            for (size_t i = 0; i < 4; ++i) {
                features["memory_" + std::string(pattern) + "_" + std::to_string(i)] = values[i];
            }
        }
        
        // Add advanced features
        features["memory_access_pattern_score"] = calculateMemoryAccessScore(node);
        features["computation_intensity"] = calculateComputationIntensity(node);
        features["data_locality_score"] = calculateDataLocalityScore(node);

        // Create fixed-length feature vector
        std::unordered_map<std::string, double> ordered_features;
        for (const auto& key : feature_names) {
            ordered_features[key] = features.count(key) ? features[key] : 0.0;
        }
        std::cout << "Extracted " << ordered_features.size() << " features for node" << std::endl;
        return ordered_features;
    }
};

// Enhanced Scalar feature extraction
struct ScalarFeatures {
    std::vector<std::string> feature_names;
    std::vector<std::string> skewed_features;
    std::vector<std::string> dropped_features;

    std::unordered_map<std::string, double> extract(const json& json_data) {
        std::cout << "Extracting scalar features" << std::endl;
        std::unordered_map<std::string, double> features;

        bool found_global = false;
        for (const auto& child : json_data["children"]) {
            if (child["name"] == "Global Features") {
                features["execution_time_ms"] = child["execution_time_ms"].get<double>();
                found_global = true;
                break;
            }
        }
        if (!found_global) {
            std::cout << "Warning: Global Features node not found" << std::endl;
        }

        std::vector<json> nodes;
        for (const auto& child : json_data["children"]) {
            if (child["name"] != "Global Features") {
                nodes.push_back(child);
            }
        }
        std::cout << "Number of nodes: " << nodes.size() << std::endl;

        double node_count = nodes.size();
        double scheduling_count = 0, total_parallelism = 0, total_bytes_at_production = 0, total_vectors = 0;
        double points_computed_total = 0, bytes_at_realization = 0, working_set = 0, bytes_at_root = 0;
        double unique_bytes_read_per_realization = 0, bytes_at_task = 0;
        double total_inner_parallelism = 0, total_outer_parallelism = 0;
        double total_vector_size = 0, total_num_scalars = 0;

        for (const auto& node : nodes) {
            if (node.contains("scheduling")) {
                auto sched = node["scheduling"];
                scheduling_count += (sched["num_realizations"].get<double>() + sched["num_productions"].get<double>());
                total_inner_parallelism += sched["inner_parallelism"].get<double>();
                total_outer_parallelism += sched["outer_parallelism"].get<double>();
                total_parallelism += (sched["inner_parallelism"].get<double>() + sched["outer_parallelism"].get<double>());
                total_bytes_at_production += sched["bytes_at_production"].get<double>();
                total_vectors += sched["num_vectors"].get<double>();
                total_vector_size += sched["vector_size"].get<double>();
                total_num_scalars += sched["num_scalars"].get<double>();
                points_computed_total += sched["points_computed_total"].get<double>();
                bytes_at_realization += sched["bytes_at_realization"].get<double>();
                working_set += sched["working_set"].get<double>();
                bytes_at_root += sched["bytes_at_root"].get<double>();
                unique_bytes_read_per_realization += sched["unique_bytes_read_per_realization"].get<double>();
                bytes_at_task += sched["bytes_at_task"].get<double>();
            }
        }

        // Basic features
        features["total_parallelism"] = node_count > 0 ? total_parallelism / node_count : 0.0;
        features["inner_parallelism_avg"] = node_count > 0 ? total_inner_parallelism / node_count : 0.0;
        features["outer_parallelism_avg"] = node_count > 0 ? total_outer_parallelism / node_count : 0.0;
        features["parallelism_ratio"] = total_inner_parallelism > 0 ? total_outer_parallelism / total_inner_parallelism : 0.0;
        features["scheduling_count"] = scheduling_count;
        features["total_bytes_at_production"] = total_bytes_at_production;
        features["total_vectors"] = total_vectors;
        features["avg_vector_size"] = total_vectors > 0 ? total_vector_size / total_vectors : 0.0;
        features["vectorization_ratio"] = (total_vectors * total_vector_size) > 0 ? 
                                         total_num_scalars / (total_vectors * total_vector_size) : 0.0;
        
        // Efficiency metrics
        features["computation_efficiency"] = bytes_at_realization > 0 ? points_computed_total / bytes_at_realization : 0.0;
        features["memory_pressure"] = bytes_at_root > 0 ? working_set / bytes_at_root : 0.0;
        features["memory_utilization_ratio"] = bytes_at_task > 0 ? unique_bytes_read_per_realization / bytes_at_task : 0.0;
        features["bytes_processing_rate"] = features["execution_time_ms"] > 0 ? bytes_at_realization / features["execution_time_ms"] : 0.0;
        features["bytes_per_parallelism"] = total_parallelism > 0 ? bytes_at_task / total_parallelism : 0.0;
        features["bytes_per_vector"] = total_vectors > 0 ? bytes_at_realization / total_vectors : 0.0;
        
        // Graph structure metrics
        features["nodes_count"] = node_count;
        double edges_count = 0;
        double max_depth = 0;
        double sum_depth = 0;
        
        // Calculate tree depth metrics
        std::function<double(const json&, double)> calculate_depth = [&](const json& node, double current_depth) -> double {
            double max_child_depth = current_depth;
            if (node.contains("children") && node["children"].is_array()) {
                for (const auto& child : node["children"]) {
                    max_child_depth = std::max(max_child_depth, calculate_depth(child, current_depth + 1));
                }
            }
            return max_child_depth;
        };
        
        max_depth = calculate_depth(json_data, 0);
        features["max_tree_depth"] = max_depth;
        
        // Calculate branching factor and edge count
        for (const auto& node : nodes) {
            if (node.contains("children")) {
                double children_count = node["children"].size();
                edges_count += children_count;
                if (children_count > 0) {
                    features["max_branching_factor"] = std::max(features["max_branching_factor"], children_count);
                }
            }
        }
        
        features["edges_count"] = edges_count;
        features["node_edge_ratio"] = edges_count > 0 ? node_count / edges_count : node_count;
        features["avg_branching_factor"] = node_count > 0 ? edges_count / node_count : 0.0;
        features["nodes_per_schedule"] = scheduling_count > 0 ? node_count / scheduling_count : node_count;
        features["tree_density"] = max_depth > 0 ? node_count / max_depth : node_count;

        // Operation diversity metrics
        std::set<std::string> ops;
        std::unordered_map<std::string, double> op_counts;
        double total_ops = 0.0;
        
        for (const auto& node : nodes) {
            if (node.contains("op_histogram")) {
                for (const auto& [op, count] : node["op_histogram"].items()) {
                    double op_count = count.get<double>();
                    if (op_count > 0) {
                        ops.insert(op);
                        op_counts[op] += op_count;
                        total_ops += op_count;
                    }
                }
            }
        }
        
        features["op_diversity"] = ops.size();
        features["op_density"] = node_count > 0 ? total_ops / node_count : 0.0;
        
        // Calculate operation entropy (measure of diversity)
        double op_entropy = 0.0;
        if (total_ops > 0) {
            for (const auto& [op, count] : op_counts) {
                double p = count / total_ops;
                if (p > 0) {
                    op_entropy -= p * std::log2(p);
                }
            }
        }
        features["op_entropy"] = op_entropy;

        // Apply log transformation to skewed features
        for (const auto& feature : skewed_features) {
            if (features.count(feature)) {
                features["log_" + feature] = std::log1p(features[feature]);
                features.erase(feature);
            }
        }

        // Create final feature map with ordered features
        std::unordered_map<std::string, double> ordered_features;
        for (const auto& key : feature_names) {
            if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                ordered_features[key] = features.count(key) ? features[key] : 0.0;
            }
        }
        return ordered_features;
    }
};

// Model ensemble for improved prediction accuracy
class ModelEnsemble {
private:
    std::vector<torch::jit::script::Module> models;
    torch::Device device;
    
public:
    ModelEnsemble(const std::vector<std::string>& model_paths, torch::Device device) : device(device) {
        for (const auto& path : model_paths) {
            try {
                torch::jit::script::Module model = torch::jit::load(path);
                model.eval();
                model.to(device);
                models.push_back(model);
                std::cout << "Loaded model: " << path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error loading model " << path << ": " << e.what() << std::endl;
            }
        }
        
        if (models.empty()) {
            throw std::runtime_error("No models were successfully loaded");
        }
        
        std::cout << "Ensemble created with " << models.size() << " models" << std::endl;
    }
    
    double predict(const torch::Tensor& seq_tensor, const torch::Tensor& scalar_tensor) {
        if (models.empty()) {
            throw std::runtime_error("No models available for prediction");
        }
        
        std::vector<double> predictions;
        for (auto& model : models) {
            std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
            auto output = model.forward(inputs).toTensor();
            predictions.push_back(output.item<double>());
        }
        
        // Calculate mean and standard deviation
        double sum = std::accumulate(predictions.begin(), predictions.end(), 0.0);
        double mean = sum / predictions.size();
        
        double sq_sum = std::inner_product(predictions.begin(), predictions.end(), predictions.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / predictions.size() - mean * mean);
        
        std::cout << "Ensemble predictions: mean=" << mean << ", stdev=" << stdev << std::endl;
        
        // Remove outliers (optional)
        if (predictions.size() > 3) {
            std::vector<double> filtered_predictions;
            for (double pred : predictions) {
                if (std::abs(pred - mean) <= 2.0 * stdev) {
                    filtered_predictions.push_back(pred);
                }
            }
            
            if (!filtered_predictions.empty()) {
                double filtered_sum = std::accumulate(filtered_predictions.begin(), filtered_predictions.end(), 0.0);
                mean = filtered_sum / filtered_predictions.size();
                std::cout << "Filtered ensemble prediction: " << mean << " (removed " 
                          << predictions.size() - filtered_predictions.size() << " outliers)" << std::endl;
            }
        }
        
        return mean;
    }
    
    size_t size() const {
        return models.size();
    }
};

// Main function
int main(int argc, char* argv[]) {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::string input_file_path = "tree_representation.json";
        if (argc > 1) {
            input_file_path = argv[1];
        }
        std::cout << "Input file: " << input_file_path << std::endl;

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

        RobustScaler scaler_node, scaler_scalar, scaler_y;
        scaler_node.load("scaler_node_params.json");
        scaler_scalar.load("scaler_scalar_params.json");
        scaler_y.load("scaler_y_params.json");

        if (scaler_node.center.size() != seq_input_size || scaler_scalar.center.size() != scalar_input_size) {
            throw std::runtime_error("Scaler dimensions do not match input sizes: node_center=" + 
                                     std::to_string(scaler_node.center.size()) + ", scalar_center=" + 
                                     std::to_string(scaler_scalar.center.size()));
        }

        std::ifstream input_file(input_file_path);
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open " + input_file_path);
        }
        json json_data;
        input_file >> json_data;
        std::cout << "Input JSON loaded" << std::endl;

        NodeFeatures node_extractor;
        node_extractor.feature_names = node_features;
        std::vector<std::vector<double>> node_sequences;

        auto traverse_nodes = [&](const json& node, auto&& traverse_nodes) -> void {
            auto features = node_extractor.extract(node);
            std::vector<double> feature_vec;
            for (const auto& key : node_features) {
                feature_vec.push_back(features[key]);
            }
            node_sequences.push_back(feature_vec);
            
            // Safely traverse children
            if (node.contains("children") && node["children"].is_array()) {
                for (const auto& child : node["children"]) {
                    traverse_nodes(child, traverse_nodes);
                }
            }
        };
        traverse_nodes(json_data, traverse_nodes);
        std::cout << "Extracted " << node_sequences.size() << " node sequences" << std::endl;

        std::vector<std::vector<double>> scaled_node_sequences;
        for (const auto& node : node_sequences) {
            auto scaled = scaler_node.transform(node);
            scaled_node_sequences.push_back(scaled);
        }

        // Determine if CUDA is available and set device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "CUDA is available, using GPU" << std::endl;
        } else {
            std::cout << "CUDA is not available, using CPU" << std::endl;
        }

        // Create tensors with double precision
        torch::Tensor seq_tensor;
        if (scaled_node_sequences.empty()) {
            throw std::runtime_error("No nodes extracted from JSON");
        }
        
        std::vector<double> padded_data(max_sequence_length * seq_input_size, 0.0);
        size_t nodes_to_copy = std::min(scaled_node_sequences.size(), static_cast<size_t>(max_sequence_length));
        for (size_t i = 0; i < nodes_to_copy; ++i) {
            for (size_t j = 0; j < seq_input_size; ++j) {
                padded_data[i * seq_input_size + j] = scaled_node_sequences[i][j];
            }
        }
        
        // Create tensors with double precision directly on the correct device
        seq_tensor = torch::from_blob(
            padded_data.data(),
            {1, max_sequence_length, seq_input_size},
            torch::kDouble
        ).clone().to(device);

        std::cout << "Sequence tensor created on " << (device.is_cuda() ? "CUDA" : "CPU") 
                  << " device: shape=[1, " << max_sequence_length << ", " << seq_input_size << "]" << std::endl;

        ScalarFeatures scalar_extractor;
        scalar_extractor.feature_names = scalar_features;
        scalar_extractor.skewed_features = skewed_features;
        scalar_extractor.dropped_features = dropped_features;

        auto scalar_features_map = scalar_extractor.extract(json_data);
        std::vector<double> scalar_vec;
        for (const auto& key : scalar_features) {
            if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                scalar_vec.push_back(scalar_features_map[key]);
            }
        }
        auto scaled_scalar = scaler_scalar.transform(scalar_vec);
        
        // Create scalar tensor with double precision
        torch::Tensor scalar_tensor = torch::from_blob(
            scaled_scalar.data(),
            {1, scalar_input_size},
            torch::kDouble
        ).clone().to(device);
        
        std::cout << "Scalar tensor created on " << (device.is_cuda() ? "CUDA" : "CPU") 
                  << " device: shape=[1, " << scalar_input_size << "]" << std::endl;

        // Load ensemble of models for better prediction
        std::vector<std::string> model_paths;
        
        // Check if ensemble models exist, otherwise fall back to single model
        std::vector<std::string> potential_models = {
            "recursive_model.pt",
            "recursive_model_v2.pt",
            "recursive_model_v3.pt"
        };
        
        for (const auto& path : potential_models) {
            std::ifstream file(path);
            if (file.good()) {
                model_paths.push_back(path);
            }
        }
        
        double scaled_output;
        if (model_paths.size() > 1) {
            // Use ensemble prediction
            ModelEnsemble ensemble(model_paths, device);
            scaled_output = ensemble.predict(seq_tensor, scalar_tensor);
            std::cout << "Ensemble inference completed with " << ensemble.size() << " models" << std::endl;
        } else {
            // Fall back to single model
            torch::jit::script::Module model;
            model = torch::jit::load("recursive_model.pt");
            model.eval();
            model.to(device);
            std::cout << "Single model loaded" << std::endl;
            
            std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
            auto output = model.forward(inputs).toTensor();
            scaled_output = output.item<double>();
            std::cout << "Single model inference completed" << std::endl;
        }
        
        std::cout << "Scaled output: " << scaled_output << std::endl;

        double log_output = scaler_y.inverse_transform(scaled_output, 0);
        double execution_time_ms = std::expm1(log_output);
        execution_time_ms = std::max(0.0, execution_time_ms);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        std::cout << "Predicted execution time: " << execution_time_ms << " ms" << std::endl;
        std::cout << "Prediction completed in " << duration << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
