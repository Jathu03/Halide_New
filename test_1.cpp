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
                     op_histogram[op] = count.get<float>();
                 }
             }
             for (const auto& op : {
                 "add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "or", "and", "not",
                 "min", "max", "constant", "variable", "funccall", "imagecall", "externcall", "let", "param"
             }) {
                 features["op_" + std::string(op)] = op_histogram[op];
             }

             // Memory patterns
             std::unordered_map<std::string, std::vector<float>> memory_patterns;
             for (const auto& pattern : {"Pointwise", "Transpose", "Broadcast", "Slice"}) {
                 memory_patterns[pattern] = {0.0f, 0.0f, 0.0f, 0.0f};
             }
             if (node.contains("memory_patterns")) {
                 for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                     std::vector<float> vals(4, 0.0f);
                     for (size_t i = 0; i < std::min<size_t>(values.size(), 4); ++i) {
                         vals[i] = values[i].get<float>();
                     }
                     memory_patterns[pattern] = vals;
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
                 ordered_features[key] = features[key];
             }
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

             // Global features
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

             // Collect nodes (excluding Global Features)
             std::vector<json> nodes;
             for (const auto& child : json_data["children"]) {
                 if (child["name"] != "Global Features") {
                     nodes.push_back(child);
                 }
             }
             std::cout << "Number of nodes: " << nodes.size() << std::endl;

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

             // Op diversity
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

             // Apply log transformation for skewed features
             for (const auto& feature : skewed_features) {
                 if (features.count(feature)) {
                     features["log_" + feature] = std::log1p(features[feature]);
                     features.erase(feature);
                 }
             }

             // Filter out dropped features and create ordered feature vector
             std::unordered_map<std::string, float> ordered_features;
             for (const auto& key : feature_names) {
                 if (std::find(dropped_features.begin(), dropped_features.end(), key) == dropped_features.end()) {
                     ordered_features[key] = features[key];
                 }
             }
             return ordered_features;
         }
     };

     int main(int argc, char* argv[]) {
         try {
             // Handle command-line argument
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

             // Verify scaler dimensions
             if (scaler_node.center.size() != seq_input_size || scaler_scalar.center.size() != scalar_input_size) {
                 throw std::runtime_error("Scaler dimensions do not match input sizes: node_center=" + 
                                          std::to_string(scaler_node.center.size()) + ", scalar_center=" + 
                                          std::to_string(scaler_scalar.center.size()));
             }

             // Load input JSON
             std::ifstream input_file(input_file_path);
             if (!input_file.is_open()) {
                 throw std::runtime_error("Failed to open " + input_file_path);
             }
             json json_data;
             input_file >> json_data;
             std::cout << "Input JSON loaded" << std::endl;

             // Extract node features
             NodeFeatures node_extractor;
             node_extractor.feature_names = node_features;
             std::vector<std::vector<float>> node_sequences;

             auto traverse_nodes = [&](const json& node, auto&& traverse_nodes) -> void {
                 auto features = node_extractor.extract(node);
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
             std::cout << "Extracted " << node_sequences.size() << " node sequences" << std::endl;

             // Scale node features
             std::vector<std::vector<float>> scaled_node_sequences;
             for (const auto& node : node_sequences) {
                 auto scaled = scaler_node.transform(node);
                 scaled_node_sequences.push_back(scaled);
             }

             // Pad sequences
             torch::Tensor seq_tensor;
             if (scaled_node_sequences.empty()) {
                 throw std::runtime_error("No nodes extracted from JSON");
             }
             std::vector<float> padded_data(max_sequence_length * seq_input_size, 0.0f);
             size_t nodes_to_copy = std::min(scaled_node_sequences.size(), static_cast<size_t>(max_sequence_length));
             for (size_t i = 0; i < nodes_to_copy; ++i) {
                 for (size_t j = 0; j < seq_input_size; ++j) {
                     padded_data[i * seq_input_size + j] = scaled_node_sequences[i][j];
                 }
             }
             seq_tensor = torch::from_blob(
                 padded_data.data(),
                 {1, max_sequence_length, seq_input_size},
                 torch::kFloat
             );
             std::cout << "Sequence tensor created: shape=[1, " << max_sequence_length << ", " << seq_input_size << "]" << std::endl;

             // Extract and scale scalar features
             ScalarFeatures scalar_extractor;
             scalar_extractor.feature_names = scalar_features;
             scalar_extractor.skewed_features = skewed_features;
             scalar_extractor.dropped_features = dropped_features;

             auto scalar_features_map = scalar_extractor.extract(json_data);
             std::vector<float> scalar_vec;
             for (const auto& key : scalar_features) {
                 scalar_vec.push_back(scalar_features_map[key]);
             }
             auto scaled_scalar = scaler_scalar.transform(scalar_vec);
             torch::Tensor scalar_tensor = torch::from_blob(
                 scaled_scalar.data(),
                 {1, scalar_input_size},
                 torch::kFloat
             );
             std::cout << "Scalar tensor created: shape=[1, " << scalar_input_size << "]" << std::endl;

             // Load model
             torch::jit::script::Module model;
             model = torch::jit::load("recursive_model.pt");
             model.eval();
             std::cout << "Model loaded" << std::endl;

             // Perform inference
             std::vector<torch::jit::IValue> inputs = {seq_tensor, scalar_tensor};
             auto output = model.forward(inputs).toTensor();
             float scaled_output = output.item<float>();
             std::cout << "Inference completed: scaled_output=" << scaled_output << std::endl;

             // Postprocess output
             float log_output = scaler_y.inverse_transform(scaled_output, 0);
             float execution_time_ms = std::expm1(log_output);
             execution_time_ms = std::max(0.0f, execution_time_ms);

             std::cout << "Predicted execution time: " << execution_time_ms << " ms" << std::endl;

         } catch (const std::exception& e) {
             std::cerr << "Error: " << e.what() << std::endl;
             return 1;
         }

         return 0;
     }
