#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>

using json = nlohmann::json;

// Utility function to load JSON from a file
json load_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    json j;
    file >> j;
    return j;
}

// Function to extract execution time (simplified version of Python's get_execution_time)
double get_execution_time(const json& data) {
    if (!data.contains("scheduling_data")) {
        throw std::runtime_error("No 'scheduling_data' in JSON");
    }
    for (const auto& item : data["scheduling_data"]) {
        if (item.is_object() && item.contains("name") && item["name"] == "total_execution_time_ms") {
            if (item.contains("value") && !item["value"].is_null()) {
                return item["value"].get<double>();
            }
        }
    }
    // Fallback: return the last value in scheduling_data
    auto schedules = data["scheduling_data"];
    if (!schedules.empty() && schedules.back().contains("value")) {
        return schedules.back()["value"].get<double>();
    }
    throw std::runtime_error("No valid execution time found");
}

// Function to extract features from a JSON file (replicates extract_features_from_file from Python)
std::map<std::string, double> extract_features_from_file(const std::string& file_path) {
    json data = load_json(file_path);
    double execution_time = get_execution_time(data);
    
    std::vector<json> nodes_features;
    std::vector<json> edges_features;
    json programming_details;
    if (data.contains("programming_details")) {
        programming_details = data["programming_details"];
    } else {
        throw std::runtime_error("No 'programming_details' in JSON");
    }
    
    // Extract nodes
    if (programming_details.contains("Nodes")) {
        for (const auto& node : programming_details["Nodes"]) {
            json node_feature;
            node_feature["Name"] = node.value("Name", "");
            if (node.contains("Details") && node["Details"].contains("Op histogram")) {
                for (const auto& op_line : node["Details"]["Op histogram"]) {
                    std::string line = op_line.get<std::string>();
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        std::string op_name = line.substr(0, pos);
                        int op_count = std::stoi(line.substr(pos + 1));
                        node_feature["op_" + op_name] = op_count;
                    }
                }
            }
            nodes_features.push_back(node_feature);
        }
    }
    
    // Extract edges
    if (programming_details.contains("Edges")) {
        for (const auto& edge : programming_details["Edges"]) {
            json edge_feature;
            edge_feature["From"] = edge.value("From", "");
            edge_feature["To"] = edge.value("To", "");
            edge_feature["Name"] = edge.value("Name", "");
            edges_features.push_back(edge_feature);
        }
    }
    
    // Extract scheduling features
    std::vector<json> scheduling_features;
    json scheduling_data;
    if (data.contains("scheduling_data")) {
        scheduling_data = data["scheduling_data"];
    } else if (programming_details.contains("Schedules")) { // Fixed: Added missing closing quotation mark
        scheduling_data = programming_details["Schedules"];
    }
    if (!scheduling_data.is_null()) {
        for (const auto& sched : scheduling_data) {
            json sched_feature;
            sched_feature["Name"] = sched.value("Name", "");
            if (sched.contains("Details") && sched["Details"].contains("scheduling_feature")) {
                for (auto& [key, value] : sched["Details"]["scheduling_feature"].items()) {
                    if (value.is_number()) {
                        sched_feature[key] = value.get<double>();
                    }
                }
            }
            scheduling_features.push_back(sched_feature);
        }
    }
    
    // Compute features
    std::map<std::string, double> features;
    features["execution_time"] = execution_time;
    features["nodes_count"] = nodes_features.size();
    features["edges_count"] = edges_features.size();
    features["scheduling_count"] = scheduling_features.size();
    
    features["node_edge_ratio"] = (edges_features.size() > 0) ? nodes_features.size() / static_cast<double>(edges_features.size()) : 0;
    
    std::map<std::string, double> op_counts;
    for (const auto& node : nodes_features) {
        for (auto& [key, value] : node.items()) {
            if (key.find("op_") == 0 && value.is_number()) {
                op_counts[key] += value.get<double>();
            }
        }
    }
    features.insert(op_counts.begin(), op_counts.end());
    
    if (!scheduling_features.empty()) {
        std::vector<std::string> important_metrics = {
            "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
            "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
            "num_scalars", "num_vectors", "points_computed_total", "working_set"
        };
        for (const auto& metric : important_metrics) {
            if (scheduling_features[0].contains(metric)) {
                features["sched_" + metric] = scheduling_features[0][metric].get<double>();
            }
        }
        
        double total_bytes_at_production = 0;
        double total_vectors = 0;
        double total_parallelism = 0;
        for (const auto& sf : scheduling_features) {
            total_bytes_at_production += sf.value("bytes_at_production", 0.0);
            total_vectors += sf.value("num_vectors", 0.0);
            total_parallelism += sf.value("inner_parallelism", 0.0) * sf.value("outer_parallelism", 1.0);
        }
        features["total_bytes_at_production"] = total_bytes_at_production;
        features["total_vectors"] = total_vectors;
        features["total_parallelism"] = total_parallelism;
        
        if (total_vectors > 0) {
            features["bytes_per_vector"] = total_bytes_at_production / total_vectors;
        }
        
        if (scheduling_features[0].contains("working_set") && scheduling_features[0].contains("bytes_at_production")) {
            double bytes = scheduling_features[0]["bytes_at_production"].get<double>();
            features["memory_pressure"] = (bytes > 0) ? scheduling_features[0]["working_set"].get<double>() / bytes : 0;
        }
    }
    
    if (!nodes_features.empty()) {
        double total_ops = 0;
        for (const auto& [key, value] : op_counts) {
            total_ops += value;
        }
        features["avg_ops_per_node"] = total_ops / nodes_features.size();
        features["op_diversity"] = op_counts.size() / static_cast<double>(nodes_features.size());
    }
    
    return features; // Added return statement to match function signature
}

// Scale features using saved scaler parameters
std::vector<double> scale_features(const std::map<std::string, double>& features,
                                  const std::vector<std::string>& feature_names,
                                  const std::vector<double>& means,
                                  const std::vector<double>& scales) {
    std::vector<double> scaled_features;
    for (size_t i = 0; i < feature_names.size(); ++i) {
        auto it = features.find(feature_names[i]);
        double value = (it != features.end()) ? it->second : 0.0;
        double scaled_value = (value - means[i]) / (scales[i] + 1e-8); // Avoid division by zero
        scaled_features.push_back(scaled_value);
    }
    return scaled_features;
}

// Create input tensor for the model
torch::Tensor create_input_tensor(const std::vector<double>& scaled_features) {
    std::vector<float> float_features(scaled_features.begin(), scaled_features.end());
    return torch::tensor(float_features).view({1, 1, static_cast<long>(float_features.size())});
}

int main() {
    // Specify the JSON file path directly
    std::string file_path = "/synthetic_data/program_50001/0_0.json";

    try {
        // Load scaler parameters
        json scaler_X_data = load_json("scaler_X.json");
        json scaler_y_data = load_json("scaler_y.json");
        
        std::vector<std::string> feature_names = scaler_X_data["feature_names"].get<std::vector<std::string>>();
        std::vector<double> means = scaler_X_data["means"].get<std::vector<double>>();
        std::vector<double> scales = scaler_X_data["scales"].get<std::vector<double>>();
        
        double y_mean = scaler_y_data["mean"].get<double>();
        double y_scale = scaler_y_data["scale"].get<double>();
        bool is_log_transformed = scaler_y_data["is_log_transformed"].get<bool>();
        
        // Extract features from the JSON file
        std::cout << "Extracting features from: " << file_path << std::endl;
        auto features = extract_features_from_file(file_path);
        
        // Scale features
        auto scaled_features = scale_features(features, feature_names, means, scales);
        
        // Create input tensor
        torch::Tensor input_tensor = create_input_tensor(scaled_features);
        
        // Load the model
        torch::jit::script::Module model = torch::jit::load("lstm_model.pt");
        model.eval();
        
        // Run inference
        std::vector<torch::jit::IValue> inputs = {input_tensor};
        torch::Tensor output = model.forward(inputs).toTensor();
        
        // Inverse scale the output
        double predicted_scaled = output.item<double>();
        double predicted = (predicted_scaled * y_scale) + y_mean;
        if (is_log_transformed) {
            predicted = std::expm1(predicted); // Inverse of log1p
        }
        
        std::cout << "Predicted execution time for " << file_path << ": " << predicted << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
