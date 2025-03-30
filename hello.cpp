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
    } else if (programming_details.contains("Schedules
