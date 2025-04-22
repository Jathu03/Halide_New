#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>

using json = nlohmann::json;

// Function to get execution time from JSON (equivalent to Python's get_execution_time)
double get_execution_time(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: File " << file_path << " not found\n";
        return -1.0;
    }

    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid JSON format in " << file_path << ": " << e.what() << "\n";
        return -1.0;
    }

    if (!data.contains("programming_details")) {
        std::cerr << "Error: 'programming_details' key not found in " << file_path << "\n";
        return -1.0;
    }

    if (!data.contains("scheduling_data")) {
        std::cerr << "Error: 'scheduling_data' key not found in " << file_path << "\n";
        return -1.0;
    }

    auto schedules = data["scheduling_data"];
    for (const auto& item : schedules) {
        if (item.is_object() && item.contains("name") && item["name"] == "total_execution_time_ms") {
            if (item.contains("value") && !item["value"].is_null()) {
                double execution_time = item["value"].get<double>();
                std::cout << "Extracted execution time for " << file_path << ": " << execution_time << " ms\n";
                return execution_time;
            }
        }
    }

    if (!schedules.empty() && schedules.back().is_object() && schedules.back().contains("value")) {
        double execution_time = schedules.back()["value"].get<double>();
        std::cout << "Warning: 'total_execution_time_ms' not found, using last schedule value: " << execution_time << " ms\n";
        return execution_time;
    }

    std::cerr << "Error: No valid execution time found in " << file_path << "\n";
    return -1.0;
}

// Function to extract features from JSON (equivalent to Python's extract_features_from_file)
std::unordered_map<std::string, double> extract_features_from_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: File " << file_path << " not found\n";
        return {};
    }

    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid JSON format in " << file_path << ": " << e.what() << "\n";
        return {};
    }

    double execution_time = get_execution_time(file_path);
    if (execution_time < 0) {
        return {};
    }

    execution_time = std::clamp(execution_time, 1.0, 10000.0);

    std::vector<std::unordered_map<std::string, double>> nodes_features;
    std::vector<std::unordered_map<std::string, std::string>> edges_features;
    auto programming_details = data.value("programming_details", json({}));

    if (programming_details.contains("Nodes")) {
        for (const auto& node : programming_details["Nodes"]) {
            std::unordered_map<std::string, double> node_feature;
            if (node.contains("Details") && node["Details"].contains("Op histogram")) {
                for (const auto& op_line : node["Details"]["Op histogram"]) {
                    std::string op_line_str = op_line.get<std::string>();
                    size_t pos = op_line_str.find(':');
                    if (pos != std::string::npos) {
                        std::string op_name = op_line_str.substr(0, pos);
                        std::string op_count_str = op_line_str.substr(pos + 1);
                        op_name.erase(op_name.find_last_not_of(" \n\r\t") + 1);
                        op_count_str.erase(0, op_count_str.find_first_not_of(" \n\r\t"));
                        int op_count = std::stoi(op_count_str);
                        node_feature["op_" + op_name] = op_count;
                    }
                }
            }
            nodes_features.push_back(node_feature);
        }
    }

    if (programming_details.contains("Edges")) {
        for (const auto& edge : programming_details["Edges"]) {
            std::unordered_map<std::string, std::string> edge_feature;
            edge_feature["From"] = edge.value("From", "");
            edge_feature["To"] = edge.value("To", "");
            edge_feature["Name"] = edge.value("Name", "");
            edges_features.push_back(edge_feature);
        }
    }

    std::vector<std::unordered_map<std::string, double>> scheduling_features;
    auto scheduling_data = data.value("scheduling_data", json::array());
    if (scheduling_data.empty() && programming_details.contains("Schedules")) {
        scheduling_data = programming_details["Schedules"];
    }

    for (const auto& sched : scheduling_data) {
        std::unordered_map<std::string, double> sched_feature;
        if (sched.contains("Details") && sched["Details"].contains("scheduling_feature")) {
            for (const auto& [key, value] : sched["Details"]["scheduling_feature"].items()) {
                if (value.is_number()) {
                    sched_feature[key] = value.get<double>();
                }
            }
        }
        scheduling_features.push_back(sched_feature);
    }

    std::unordered_map<std::string, double> features;
    features["execution_time"] = execution_time;
    features["nodes_count"] = nodes_features.size();
    features["edges_count"] = edges_features.size();
    features["scheduling_count"] = scheduling_features.size();

    if (!nodes_features.empty() && !edges_features.empty()) {
        features["node_edge_ratio"] = static_cast<double>(nodes_features.size()) / edges_features.size();
    } else {
        features["node_edge_ratio"] = 0.0;
    }

    std::unordered_map<std::string, double> op_counts;
    for (const auto& node : nodes_features) {
        for (const auto& [key, value] : node) {
            if (key.find("op_") == 0) {
                op_counts[key] += value;
            }
        }
    }
    for (const auto& [key, value] : op_counts) {
        features[key] = value;
    }

    if (!scheduling_features.empty()) {
        std::vector<std::string> important_metrics = {
            "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
            "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
            "num_scalars", "num_vectors", "points_computed_total", "working_set"
        };

        if (!scheduling_features.empty()) {
            for (const auto& metric : important_metrics) {
                if (scheduling_features[0].count(metric)) {
                    features["sched_" + metric] = scheduling_features[0][metric];
                }
            }
        }

        double total_bytes_at_production = 0.0;
        double total_vectors = 0.0;
        double total_parallelism = 0.0;
        double points_computed_total = 0.0;
        double working_set = 0.0;
        double total_inner_parallelism = 0.0;

        for (const auto& sf : scheduling_features) {
            total_bytes_at_production += sf.count("bytes_at_production") ? sf.at("bytes_at_production") : 0.0;
            total_vectors += sf.count("num_vectors") ? sf.at("num_vectors") : 0.0;
            double inner = sf.count("inner_parallelism") ? sf.at("inner_parallelism") : 0.0;
            double outer = sf.count("outer_parallelism") ? sf.at("outer_parallelism") : 1.0;
            total_parallelism += inner * outer;
            points_computed_total += sf.count("points_computed_total") ? sf.at("points_computed_total") : 0.0;
            working_set += sf.count("working_set") ? sf.at("working_set") : 0.0;
            total_inner_parallelism += inner;
        }

        double comp_efficiency = (total_bytes_at_production != 0.0) ? points_computed_total / std::max(total_bytes_at_production, 1e-4) : 0.0;
        double bytes_processing_rate = (execution_time != 0.0) ? total_bytes_at_production / std::max(execution_time, 1e-4) : 0.0;
        double mem_util_ratio = (total_bytes_at_production != 0.0) ? working_set / std::max(total_bytes_at_production, 1e-4) : 0.0;

        features["total_bytes_at_production"] = total_bytes_at_production;
        features["total_vectors"] = total_vectors;
        features["total_parallelism"] = total_parallelism;
        features["computation_efficiency"] = comp_efficiency;
        features["bytes_processing_rate"] = bytes_processing_rate;
        features["memory_utilization_ratio"] = mem_util_ratio;
        features["sched_inner_parallelism_squared"] = total_inner_parallelism * total_inner_parallelism;
        features["computation_efficiency_squared"] = comp_efficiency * comp_efficiency;
        features["comp_efficiency_total_vectors"] = comp_efficiency * total_vectors;
        features["inner_parallelism_total_parallelism"] = total_inner_parallelism * total_parallelism;

        if (total_vectors > 0) {
            features["bytes_per_vector"] = total_bytes_at_production / total_vectors;
        } else {
            features["bytes_per_vector"] = 0.0;
        }

        if (scheduling_features[0].count("working_set") && scheduling_features[0].count("bytes_at_production")) {
            features["memory_pressure"] = (scheduling_features[0]["bytes_at_production"] > 0) ?
                scheduling_features[0]["working_set"] / scheduling_features[0]["bytes_at_production"] : 0.0;
        }
    }

    if (!nodes_features.empty()) {
        size_t op_types = op_counts.size();
        double total_ops = 0.0;
        for (const auto& [_, value] : op_counts) {
            total_ops += value;
        }
        features["avg_ops_per_node"] = total_ops / nodes_features.size();
        features["op_diversity"] = static_cast<double>(op_types) / nodes_features.size();
    }

    return features;
}

// Function to apply transformations equivalent to clean_and_transform_features
std::vector<double> transform_features(const std::unordered_map<std::string, double>& features,
                                       const std::vector<std::string>& final_features,
                                       const std::vector<double>& scaler_X_mean,
                                       const std::vector<double>& scaler_X_scale) {
    std::unordered_map<std::string, double> transformed_features = features;

    // Fill missing features with 0
    for (const auto& feature : final_features) {
        if (transformed_features.find(feature) == transformed_features.end()) {
            transformed_features[feature] = 0.0;
        }
    }

    // Apply log transformations for skewed features
    std::vector<std::string> skewed_features = {
        "computation_efficiency", "bytes_processing_rate", "total_parallelism", "total_vectors", "bytes_per_vector"
    };
    for (const auto& feature : skewed_features) {
        if (transformed_features.count(feature)) {
            transformed_features["log_" + feature] = std::log1p(transformed_features[feature]);
            transformed_features.erase(feature);
        }
    }

    // Create feature vector in the order of final_features
    std::vector<double> feature_vector;
    for (const auto& feature : final_features) {
        feature_vector.push_back(transformed_features[feature]);
    }

    // Apply StandardScaler (equivalent to scaler_X.transform)
    for (size_t i = 0; i < feature_vector.size(); ++i) {
        feature_vector[i] = (feature_vector[i] - scaler_X_mean[i]) / scaler_X_scale[i];
    }

    return feature_vector;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_json_file>\n";
        return 1;
    }

    std::string input_file = argv[1];

    // Load final features
    std::ifstream features_file("final_features.json");
    if (!features_file.is_open()) {
        std::cerr << "Error: Could not open final_features.json\n";
        return 1;
    }
    json final_features_json;
    features_file >> final_features_json;
    std::vector<std::string> final_features = final_features_json.get<std::vector<std::string>>();

    // Load scaler parameters
    std::ifstream scaler_file("scaler_params.json");
    if (!scaler_file.is_open()) {
        std::cerr << "Error: Could not open scaler_params.json\n";
        return 1;
    }
    json scaler_params;
    scaler_file >> scaler_params;

    std::vector<double> scaler_X_mean = scaler_params["scaler_X_mean"].get<std::vector<double>>();
    std::vector<double> scaler_X_scale = scaler_params["scaler_X_scale"].get<std::vector<double>>();
    std::vector<double> scaler_y_center = scaler_params["scaler_y_center"].get<std::vector<double>>();
    std::vector<double> scaler_y_scale = scaler_params["scaler_y_scale"].get<std::vector<double>>();
    bool is_log_transformed = scaler_params["is_log_transformed"].get<bool>();

    // Extract features from the input JSON file
    auto features = extract_features_from_file(input_file);
    if (features.empty()) {
        std::cerr << "Error: Failed to extract features from " << input_file << "\n";
        return 1;
    }

    // Transform features
    auto feature_vector = transform_features(features, final_features, scaler_X_mean, scaler_X_scale);

    // Convert feature vector to a tensor (shape: [1, 1, input_size])
    torch::Tensor input_tensor = torch::tensor(feature_vector, torch::dtype(torch::kFloat32))
        .reshape({1, 1, static_cast<long>(feature_vector.size())});

    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("lstm_model.pt");
        module.eval();
    } catch (const std::exception& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return 1;
    }

    // Perform inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    auto output = module.forward(inputs).toTensor();

    // Reverse the scaling of the output (RobustScaler inverse_transform)
    float output_value = output.item<float>();
    float transformed_output = output_value * scaler_y_scale[0] + scaler_y_center[0];

    // Reverse log transformation if applied
    float predicted_execution_time = is_log_transformed ? std::expm1(transformed_output) : transformed_output;
    predicted_execution_time = std::max(predicted_execution_time, 1e-2f);

    std::cout << "Predicted execution time for " << input_file << ": " << predicted_execution_time << " ms\n";

    return 0;
}
