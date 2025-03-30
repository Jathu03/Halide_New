#include <torch/torch.h>          // Core tensor library (includes CUDA support)
#include <torch/script.h>         // For JIT model loading
#include <nlohmann/json.hpp>      // JSON parsing library
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using json = nlohmann::json;

// Function to get execution time from a JSON file
float get_execution_time(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: File " << file_path << " not found" << std::endl;
        return -1.0f;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    try {
        json data = json::parse(content);
        if (!data.contains("scheduling_data")) {
            std::cerr << "Error: 'scheduling_data' key not found in " << file_path << std::endl;
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
        return -1.0f; // Return invalid time if not found
    } catch (const json::exception& e) {
        std::cerr << "Error: Invalid JSON in " << file_path << ": " << e.what() << std::endl;
        return -1.0f;
    }
}

// Function to extract features from a JSON file
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
        std::cerr << "Warning: No valid execution time found in " << file_path << std::endl;
        return {};
    }

    std::vector<std::map<std::string, std::string>> nodes_features;
    std::vector<std::map<std::string, std::string>> edges_features;
    json programming_details = data.value("programming_details", json::object());

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

    if (programming_details.contains("Edges")) {
        for (const auto& edge : programming_details["Edges"]) {
            std::map<std::string, std::string> edge_feature;
            edge_feature["From"] = edge.value("From", "");
            edge_feature["To"] = edge.value("To", "");
            edge_feature["Name"] = edge.value("Name", "");
            edges_features.push_back(edge_feature);
        }
    }

    std::map<std::string, float> features;
    features["execution_time"] = execution_time;
    features["nodes_count"] = static_cast<float>(nodes_features.size());
    features["edges_count"] = static_cast<float>(edges_features.size());
    features["node_edge_ratio"] = (edges_features.size() > 0) ? 
                                  nodes_features.size() / static_cast<float>(edges_features.size()) : 0.0f;

    std::map<std::string, float> op_counts;
    for (const auto& node : nodes_features) {
        for (const auto& [key, value] : node) {
            if (key.find("op_") == 0) {
                op_counts[key] += std::stof(value);
            }
        }
    }
    features.insert(op_counts.begin(), op_counts.end());

    return features;
}

// Function to convert features to a tensor
torch::Tensor features_to_tensor(const std::map<std::string, float>& features) {
    std::vector<std::string> feature_order = {
        "nodes_count", "edges_count", "node_edge_ratio",
        "op_add", "op_mul" // Add more ops as needed based on your data
    };

    std::vector<float> feature_vec;
    for (const auto& key : feature_order) {
        feature_vec.push_back(features.count(key) ? features.at(key) : 0.0f);
    }

    // Simple normalization (replace with actual mean and std if available)
    float mean = 0.0f, std = 1.0f;
    for (auto& val : feature_vec) {
        val = (val - mean) / (std + 1e-6); // Avoid division by zero
    }

    auto tensor = torch::from_blob(feature_vec.data(), {1, static_cast<long>(feature_vec.size())});
    if (torch::cuda::is_available()) {
        return tensor.to(torch::kCUDA);
    }
    return tensor;
}

int main() {
    // Check CUDA availability
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "CUDA not available. Using CPU." << std::endl;
    }

    // Load the model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("/home/kowrisaan/jathu/Halide_New/lstm_model.pt");
        model.eval();
        if (torch::cuda::is_available()) {
            model.to(torch::kCUDA);
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Process the JSON file
    std::string file_path = "/home/kowrisaan/jathu/Halide_New/synthetic_data/program_50001/0_0.json";
    std::cout << "Processing file: " << file_path << std::endl;

    auto features = extract_features_from_file(file_path);
    if (features.empty()) {
        std::cerr << "Failed to extract features from " << file_path << std::endl;
        return -1;
    }

    // Convert features to tensor and run inference
    torch::Tensor input = features_to_tensor(features);
    std::vector<torch::jit::IValue> inputs = {input};
    torch::Tensor output;
    try {
        output = model.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }

    // Denormalize the predicted time (replace with your actual y_mean and y_std)
    float y_mean = 0.0f, y_std = 1.0f;
    float predicted_time = output.item<float>() * y_std + y_mean;
    std::cout << "Predicted execution time for " << file_path << ": " << predicted_time << " ms" << std::endl;

    return 0;
}
