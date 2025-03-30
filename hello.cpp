#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using json = nlohmann::json;

// Function to mimic Python's get_execution_time
float get_execution_time(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: File " << file_path << " not found" << std::endl;
        return -1.0f;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    try {
        json data = json::parse(content);
        if (!data.contains("programming_details")) {
            std::cerr << "Error: 'programming_details' key not found in " << file_path << std::endl;
            return -1.0f;
        }

        auto schedules = data["scheduling_data"];
        for (const auto& item : schedules) {
            if (item.is_object() && item.contains("name") && item["name"] == "total_execution_time_ms") {
                if (item.contains("value") && !item["value"].is_null()) {
                    return item["value"].get<float>();
                }
            }
        }
        std::cerr << "Warning: 'total_execution_time_ms' not found in " << file_path << std::endl;
        return schedules.back()["value"].get<float>();
    } catch (const json::exception& e) {
        std::cerr << "Error: Invalid JSON in " << file_path << ": " << e.what() << std::endl;
        return -1.0f;
    }
}

// Function to extract features from JSON file (ported from Python)
std::map<std::string, float> extract_features_from_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << file_path << std::endl;
        return {};
    }

    json data;
    try {
        file >> data;
    } catch (const json::exception& e) {
        std::cerr << "Error: JSON parse failed for " << file_path << ": " << e.what() << std::endl;
        return {};
    }

    float execution_time = get_execution_time(file_path);
    if (execution_time < 0) {
        std::cerr << "Warning: No execution time found in " << file_path << std::endl;
        return {};
    }

    std::vector<std::map<std::string, std::string>> nodes_features;
    std::vector<std::map<std::string, std::string>> edges_features;
    json programming_details;

    if (data.contains("programming_details")) {
        programming_details = data["programming_details"];

        // Nodes
        if (programming_details.contains("Nodes")) {
            for (const auto& node : programming_details["Nodes"]) {
                std::map<std::string, std::string> node_feature;
                node_feature["Name"] = node.value("Name", "");
                if (node.contains("Details") && node["Details"].contains("Op histogram")) {
                    for (const auto& op_line : node["Details"]["Op histogram"]) {
                        std::string line = op_line.get<std::string>();
                        size_t colon = line.find(':');
                        if (colon != std::string::npos) {
                            std::string op_name = line.substr(0, colon);
                            int op_count = std::stoi(line.substr(colon + 1));
                            node_feature["op_" + op_name] = std::to_string(op_count);
                        }
                    }
                }
                nodes_features.push_back(node_feature);
            }
        }

        // Edges
        if (programming_details.contains("Edges")) {
            for (const auto& edge : programming_details["Edges"]) {
                std::map<std::string, std::string> edge_feature;
                edge_feature["From"] = edge.value("From", "");
                edge_feature["To"] = edge.value("To", "");
                edge_feature["Name"] = edge.value("Name", "");
                edges_features.push_back(edge_feature);
            }
        }
    }

    // Scheduling features
    std::vector<std::map<std::string, float>> scheduling_features;
    json scheduling_data = data.value("scheduling_data", programming_details.value("Schedules", json::array()));
    for (const auto& sched : scheduling_data) {
        std::map<std::string, float> sched_feature;
        sched_feature["Name"] = sched.value("Name", 0.0f); // Default to 0 if not string
        if (sched.contains("Details") && sched["Details"].contains("scheduling_feature")) {
            for (const auto& [key, value] : sched["Details"]["scheduling_feature"].items()) {
                sched_feature[key] = value.get<float>();
            }
        }
        scheduling_features.push_back(sched_feature);
    }

    // Aggregate features
    std::map<std::string, float> features;
    features["execution_time"] = execution_time;
    features["nodes_count"] = static_cast<float>(nodes_features.size());
    features["edges_count"] = static_cast<float>(edges_features.size());
    features["scheduling_count"] = static_cast<float>(scheduling_features.size());
    features["node_edge_ratio"] = (nodes_features.size() > 0 && edges_features.size() > 0) ? 
                                  nodes_features.size() / edges_features.size() : 0.0f;

    // Operation counts
    std::map<std::string, float> op_counts;
    for (const auto& node : nodes_features) {
        for (const auto& [key, value] : node) {
            if (key.find("op_") == 0) {
                op_counts[key] += std::stof(value);
            }
        }
    }
    features.insert(op_counts.begin(), op_counts.end());

    // Scheduling metrics
    if (!scheduling_features.empty()) {
        std::vector<std::string> important_metrics = {
            "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
            "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
            "num_scalars", "num_vectors", "points_computed_total", "working_set"
        };
        for (const auto& metric : important_metrics) {
            if (scheduling_features[0].count(metric)) {
                features["sched_" + metric] = scheduling_features[0].at(metric);
            }
        }

        float total_bytes_at_production = 0.0f, total_vectors = 0.0f, total_parallelism = 0.0f;
        for (const auto& sf : scheduling_features) {
            total_bytes_at_production += sf.count("bytes_at_production") ? sf.at("bytes_at_production") : 0.0f;
            total_vectors += sf.count("num_vectors") ? sf.at("num_vectors") : 0.0f;
            total_parallelism += (sf.count("inner_parallelism") ? sf.at("inner_parallelism") : 0.0f) *
                                 (sf.count("outer_parallelism") ? sf.at("outer_parallelism") : 1.0f);
        }
        features["total_bytes_at_production"] = total_bytes_at_production;
        features["total_vectors"] = total_vectors;
        features["total_parallelism"] = total_parallelism;

        if (total_vectors > 0) {
            features["bytes_per_vector"] = total_bytes_at_production / total_vectors;
        }
        if (scheduling_features[0].count("working_set") && scheduling_features[0].count("bytes_at_production")) {
            features["memory_pressure"] = scheduling_features[0].at("working_set") / 
                                         (scheduling_features[0].at("bytes_at_production") > 0 ? 
                                          scheduling_features[0].at("bytes_at_production") : 1.0f);
        }
    }

    if (!nodes_features.empty()) {
        float total_ops = 0.0f;
        for (const auto& [key, value] : op_counts) {
            total_ops += value;
        }
        features["avg_ops_per_node"] = total_ops / nodes_features.size();
        features["op_diversity"] = static_cast<float>(op_counts.size()) / nodes_features.size();
    }

    return features;
}

// Convert features to tensor (must match Python training order)
torch::Tensor features_to_tensor(const std::map<std::string, float>& features) {
    std::vector<std::string> feature_order = {
        "nodes_count", "edges_count", "scheduling_count", "node_edge_ratio",
        "op_add", "op_mul", /* ... add all ops from your data ... */
        "sched_bytes_at_production", "sched_bytes_at_realization", "sched_bytes_at_root",
        "sched_bytes_at_task", "sched_inner_parallelism", "sched_outer_parallelism",
        "sched_num_productions", "sched_num_realizations", "sched_num_scalars",
        "sched_num_vectors", "sched_points_computed_total", "sched_working_set",
        "total_bytes_at_production", "total_vectors", "total_parallelism",
        "bytes_per_vector", "memory_pressure", "avg_ops_per_node", "op_diversity"
    };

    std::vector<float> feature_vec;
    for (const auto& key : feature_order) {
        feature_vec.push_back(features.count(key) ? features.at(key) : 0.0f);
    }

    float mean = 5.0f, std = 2.0f; // Replace with actual values
    for (auto& val : feature_vec) {
        val = (val - mean) / std;
    }

    return torch::from_blob(feature_vec.data(), {1, 1, static_cast<long>(feature_vec.size())});
}

int main() {
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("lstm_model.pt");
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    std::string file_path = "synthetic_data/program_50001/0_0.json";
    std::cout << "Processing file: " << file_path << std::endl;

    auto features = extract_features_from_file(file_path);
    if (features.empty()) {
        std::cerr << "Failed to extract features from " << file_path << std::endl;
        return -1;
    }

    torch::Tensor input = features_to_tensor(features);
    std::vector<torch::jit::IValue> inputs = {input};
    torch::Tensor output;
    try {
        output = model.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }

    float y_mean = 0.0f, y_std = 1.0f; // Replace with actual values
    float predicted_time_scaled = output.item<float>();
    float predicted_time = predicted_time_scaled * y_std + y_mean;
    if (predicted_time < 0) {
        predicted_time = std::exp(predicted_time) - 1;
    }
    std::cout << "Predicted execution time for " << file_path << ": " << predicted_time << " ms" << std::endl;

    return 0;
}
