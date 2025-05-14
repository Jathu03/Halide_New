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
#include <iomanip>
#include <chrono>

using json = nlohmann::json;
using namespace std::chrono;

// RobustScaler with improved error checking
struct RobustScaler {
    std::vector<float> center;
    std::vector<float> scale;
    std::string name;

    RobustScaler(const std::string& scaler_name = "unnamed") : name(scaler_name) {}

    void load(const std::string& params_file) {
        std::cout << "Loading " << name << " scaler from " << params_file << std::endl;
        std::ifstream file(params_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open " + params_file);
        }
        json params;
        try {
            file >> params;
            if (!params.contains("center") || !params.contains("scale")) {
                throw std::runtime_error("Missing center or scale in " + params_file);
            }
            center = params["center"].get<std::vector<float>>();
            scale = params["scale"].get<std::vector<float>>();
            
            // Verify scale doesn't contain zeros
            for (size_t i = 0; i < scale.size(); ++i) {
                if (std::abs(scale[i]) < 1e-10) {
                    std::cout << "Warning: " << name << " scale[" << i << "] is near zero, using epsilon" << std::endl;
                    scale[i] = 1e-10;
                }
            }
            
            std::cout << name << " scaler loaded: center size=" << center.size() 
                     << ", scale size=" << scale.size() << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing " + params_file + ": " + e.what());
        }
    }

    std::vector<float> transform(const std::vector<float>& input) {
        if (input.size() != center.size()) {
            throw std::runtime_error(name + " input size (" + std::to_string(input.size()) + 
                                    ") does not match scaler size (" + std::to_string(center.size()) + ")");
        }
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = (input[i] - center[i]) / (scale[i] + 1e-8);
        }
        return output;
    }

    float inverse_transform(float input, size_t index) {
        if (index >= scale.size()) {
            throw std::runtime_error(name + " index " + std::to_string(index) + 
                                    " out of bounds for scaler (size=" + std::to_string(scale.size()) + ")");
        }
        return input * scale[index] + center[index];
    }
};

// Node feature extraction
struct NodeFeatures {
    std::vector<std::string> feature_names;

    std::unordered_map<std::string, float> extract(const json& node) {
        std::cout << "Extracting features for node: " << (node.contains("name") ? node["name"].get<std::string>() : "unnamed") << std::endl;
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
        } else {
            std::cout << "Warning: Node missing scheduling field" << std::endl;
        }

        // Operation histogram
        std::unordered_map<std::string, float> op_histogram;
        if (node.contains("op_histogram")) {
            for (const auto& [op, count] : node["op_histogram"].items()) {
                try {
                    op_histogram[op] = count.get<float>();
                } catch (const std::exception& e) {
                    std::cout << "Warning: Invalid op_histogram value for " << op << ": " << e.what() << std::endl;
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
            std::cout << "Processing memory_patterns" << std::endl;
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                std::string pattern_lower = pattern;
                std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                
                if (memory_patterns.count(pattern_lower)) {
                    std::vector<float> vals(4, 0.0f);
                    
                    // Safely check if values is an array and has elements
                    if (values.is_array()) {
                        size_t array_size = values.size();
                        std::cout << "memory_patterns[" << pattern_lower << "] has " << array_size << " elements" << std::endl;
                        
                        for (size_t i = 0; i < std::min<size_t>(array_size, 4); ++i) {
                            try {
                                if (i < array_size && values[i].is_number()) {
                                    vals[i] = values[i].get<float>();
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

        // Create fixed-length feature vector
        std::unordered_map<std::string, float> ordered_features;
        for (const auto& key : feature_names) {
            ordered_features[key] = features.count(key) ? features[key] : 0.0f;
        }
        std::cout << "Extracted " << ordered_features.size() << " features for node" << std::endl;
        return ordered_features;
    }
};

// Scalar feature extraction
struct ScalarFeatures {
    std::vector<std::string> feature_names;
    std::vector<std::string> skewed_features;
    std::vector<std::string> dropped_features;

    std::unordered_map<std::string, float> extract(const json& json_data) {
        std::cout << "Extracting scalar features" << std::endl;
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
            std::cout << "Warning: Global Features node not found" << std::endl;
        }

        std::vector<json> nodes;
        for (const auto& child : json_data["children"]) {
            if (child["name"] != "Global Features") {
                nodes.push_back(child);
            }
        }
        std::cout << "Number of nodes: " << nodes.size() << std::endl;

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

// Main function
int main(int argc, char* argv[]) {
    try {
        // Start timing the whole process
        auto total_start_time = high_resolution_clock::now();
        
        std::string input_file_path = "tree_representation.json";
        if (argc > 1) {
            input_file_path = argv[1];
        }
        std::cout << "Input file: " << input_file_path << std::endl;

        // Improved error handling for files
        std::ifstream metadata_file("model_metadata.json");
        if (!metadata_file.is_open()) {
            throw std::runtime_error("Failed to open model_metadata.json - Please ensure the file exists in the working directory");
        }
        json metadata;
        try {
            metadata_file >> metadata;
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing model_metadata.json: " + std::string(e.what()));
        }

        // Validate required metadata fields
        for (const auto& field : {"max_sequence_length", "seq_input_size", "scalar_input_size", 
                                "node_features", "scalar_features"}) {
            if (!metadata.contains(field)) {
                throw std::runtime_error(std::string("Missing required field in metadata: ") + field);
            }
        }

        int max_sequence_length = metadata["max_sequence_length"].get<int>();
        int seq_input_size = metadata["seq_input_size"].get<int>();
        int scalar_input_size = metadata["scalar_input_size"].get<int>();
        std::vector<std::string> node_features = metadata["node_features"].get<std::vector<std::string>>();
        std::vector<std::string> scalar_features = metadata["scalar_features"].get<std::vector<std::string>>();
        std::vector<std::string> skewed_features = metadata.contains("skewed_features") ? 
                                                 metadata["skewed_features"].get<std::vector<std::string>>() : 
                                                 std::vector<std::string>();
        std::vector<std::string> dropped_features = metadata.contains("dropped_features") ? 
                                                  metadata["dropped_features"].get<std::vector<std::string>>() : 
                                                  std::vector<std::string>();
        
        std::cout << "Metadata loaded: max_sequence_length=" << max_sequence_length 
                 << ", seq_input_size=" << seq_input_size 
                 << ", scalar_input_size=" << scalar_input_size << std::endl;

        RobustScaler scaler_node("Node"), scaler_scalar("Scalar"), scaler_y("Output");
        scaler_node.load("scaler_node_params.json");
        scaler_scalar.load("scaler_scalar_params.json");
        scaler_y.load("scaler_y_params.json");

        if (scaler_node.center.size() != seq_input_size || scaler_scalar.center.size() != scalar_input_size) {
            throw std::runtime_error("Scaler dimensions do not match input sizes: node_center=" + 
                                     std::to_string(scaler_node.center.size()) + ", scalar_center=" + 
                                     std::to_string(scaler_scalar.center.size()) +
                                     ", expected seq_input_size=" + std::to_string(seq_input_size) +
                                     ", expected scalar_input_size=" + std::to_string(scalar_input_size));
        }

        std::ifstream input_file(input_file_path);
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open input file: " + input_file_path);
        }
        json json_data;
        try {
            input_file >> json_data;
            std::cout << "Input JSON loaded successfully" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing input JSON file: " + std::string(e.what()));
        }

        // Feature extraction timing
        auto feature_start_time = high_resolution_clock::now();

        NodeFeatures node_extractor;
        node_extractor.feature_names = node_features;
        std::vector<std::vector<float>> node_sequences;

    auto traverse_nodes = [&](const json& node, auto&& traverse_nodes) -> void {
            auto features = node_extractor.extract(node);
            std::vector<float> feature_vec;
            for (const auto& key : node_features) {
                feature_vec.push_back(features[key]);
                // For debugging specific anomalies in features
                if (std::isnan(features[key]) || std::isinf(features[key])) {
                    std::cout << "Warning: Feature '" << key << "' has invalid value: " 
                              << features[key] << " for node: " 
                              << (node.contains("name") ? node["name"].get<std::string>() : "unnamed")
                              << std::endl;
                    // Replace NaN/inf with 0 to prevent model errors
                    feature_vec.back() = 0.0f;
                }
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

        std::vector<std::vector<float>> scaled_node_sequences;
        for (const auto& node : node_sequences) {
            auto scaled = scaler_node.transform(node);
            scaled_node_sequences.push_back(scaled);
        }

        // Determine if CUDA is available and set device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            // Make sure CUDA is properly initialized
            try {
                torch::Tensor test_tensor = torch::ones({1}, torch::kFloat).to(device);
                std::cout << "CUDA is available and working, using GPU device " 
                          << torch::cuda::current_device() << std::endl;
                
                // Optionally set device properties for better performance
                c10::cuda::CUDACachingAllocator::emptyCache();
                
                // Display GPU info
                std::cout << "GPU: " << torch::cuda::get_device_name() << std::endl;
                std::cout << "CUDA capability: " << torch::cuda::get_device_capability() << std::endl;
            } catch (const c10::Error& e) {
                std::cout << "CUDA initialization failed: " << e.what() << std::endl;
                std::cout << "Falling back to CPU" << std::endl;
                device = torch::kCPU;
            }
        } else {
            std::cout << "CUDA is not available, using CPU" << std::endl;
        }

        // Load the model first so we can move it to the right device
        torch::jit::script::Module model;
        try {
            // Measure model loading time
            auto model_load_start = high_resolution_clock::now();
            model = torch::jit::load("recursive_model.pt");
            auto model_load_end = high_resolution_clock::now();
            auto model_load_duration = duration_cast<milliseconds>(model_load_end - model_load_start).count();
            std::cout << "Model loaded in " << model_load_duration << " ms" << std::endl;
            
            model.eval();
            // Move model to the selected device
            model.to(device);
            std::cout << "Model moved to " << (device.is_cuda() ? "CUDA" : "CPU") << " device" << std::endl;
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model: " + std::string(e.what()));
        }

        // Create tensors directly on the correct device
        torch::Tensor seq_tensor;
        if (scaled_node_sequences.empty()) {
            throw std::runtime_error("No nodes extracted from JSON");
        }
        
        auto tensor_creation_start = high_resolution_clock::now();
        
        // Prepare padded data with zeros
        std::vector<float> padded_data(max_sequence_length * seq_input_size, 0.0f);
        
        // Find out how many nodes we can include (limited by max_sequence_length)
        size_t nodes_to_copy = std::min(scaled_node_sequences.size(), static_cast<size_t>(max_sequence_length));
        
        // Copy the data to the padded array
        for (size_t i = 0; i < nodes_to_copy; ++i) {
            for (size_t j = 0; j < seq_input_size; ++j) {
                padded_data[i * seq_input_size + j] = scaled_node_sequences[i][j];
            }
        }
        
        // Use torch::from_blob for zero-copy initialization, then clone and move to device
        seq_tensor = torch::from_blob(
            padded_data.data(),
            {1, max_sequence_length, seq_input_size},
            torch::kFloat
        ).clone().to(device);

        auto tensor_creation_end = high_resolution_clock::now();
        auto tensor_creation_duration = duration_cast<milliseconds>(tensor_creation_end - tensor_creation_start).count();
        
        std::cout << "Sequence tensor created on " << (device.is_cuda() ? "CUDA" : "CPU") 
                  << " device in " << tensor_creation_duration << " ms: shape=[1, " 
                  << max_sequence_length << ", " << seq_input_size << "]" << std::endl;

        ScalarFeatures scalar_extractor;
        scalar_extractor.feature_names = scalar_features;
        scalar_extractor.skewed_features = skewed_features;
        scalar_extractor.dropped_features = dropped_features;

        auto scalar_features_map = scalar_extractor.extract(json_data);
        std::vector<float> scalar_vec;
        for (const auto& key : scalar_features) {
            if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                scalar_vec.push_back(scalar_features_map[key]);
            }
        }
        auto scaled_scalar = scaler_scalar.transform(scalar_vec);
        
        // Create scalar tensor directly on the correct device
        torch::Tensor scalar_tensor = torch::from_blob(
            scaled_scalar.data(),
            {1, scalar_input_size},
            torch::kFloat
        ).clone().to(device);
        
        std::cout << "Scalar tensor created on " << (device.is_cuda() ? "CUDA" : "CPU") 
                  << " device: shape=[1, " << scalar_input_size << "]" << std::endl;

        // Run inference with both inputs on the same device as the model
        auto inference_start = high_resolution_clock::now();
        
        std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
        torch::NoGradGuard no_grad; // Disable gradient calculation for inference
        
        auto output = model.forward(inputs).toTensor();
        
        // Make sure to get the value back to CPU before using item()
        float scaled_output = output.to(torch::kCPU).item<float>();
        
        auto inference_end = high_resolution_clock::now();
        auto inference_duration = duration_cast<milliseconds>(inference_end - inference_start).count();
        
        std::cout << "Inference completed in " << inference_duration << " ms, scaled_output=" << scaled_output << std::endl;

        // Post-process output
        float log_output = scaler_y.inverse_transform(scaled_output, 0);
        float execution_time_ms = std::expm1(log_output); // inverse of log1p
        execution_time_ms = std::max(0.0f, execution_time_ms);
        
        // End timing and display total time
        auto total_end_time = high_resolution_clock::now();
        auto total_duration = duration_cast<milliseconds>(total_end_time - total_start_time).count();
        
        std::cout << std::fixed << std::setprecision(2)
                  << "Predicted execution time: " << execution_time_ms << " ms" << std::endl;
        std::cout << "Total processing time: " << total_duration << " ms" << std::endl;
        
        // Optional: Calculate confidence interval or error bounds
        // This is a simple approach - more sophisticated methods would be better
        float prediction_error_margin = 0.1f; // 10% error margin
        float lower_bound = execution_time_ms * (1.0f - prediction_error_margin);
        float upper_bound = execution_time_ms * (1.0f + prediction_error_margin);
        
        std::cout << "95% Confidence interval: [" << lower_bound << " ms, " 
                  << upper_bound << " ms]" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch Error: " << e.what() << std::endl;
        // Provide more detailed error handling for common PyTorch errors
        std::string error_msg = e.what();
        if (error_msg.find("CUDA") != std::string::npos) {
            std::cerr << "This appears to be a CUDA error. Possible solutions:" << std::endl;
            std::cerr << "1. Check if CUDA is properly installed and configured" << std::endl;
            std::cerr << "2. Try running with CPU only by setting CUDA_VISIBLE_DEVICES=-1" << std::endl;
            std::cerr << "3. Make sure your GPU has enough memory for this model" << std::endl;
        } else if (error_msg.find("size mismatch") != std::string::npos || 
                  error_msg.find("shape") != std::string::npos) {
            std::cerr << "This appears to be a tensor shape/size mismatch. Possible solutions:" << std::endl;
            std::cerr << "1. Check if the model expects different input dimensions" << std::endl;
            std::cerr << "2. Make sure feature extraction is producing the correct number of features" << std::endl;
        }
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}
