#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp> // For JSON parsing
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <cmath>

// Using namespace for convenience
using json = nlohmann::json;
namespace fs = std::filesystem;

// Function to extract features from JSON (simplified version from Python)
std::vector<float> extract_features_from_json(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    json data;
    file >> data;

    // Extract execution time (for reference)
    float execution_time = 0.0;
    if (data.contains("scheduling_data")) {
        for (const auto& item : data["scheduling_data"]) {
            if (item.is_object() && item["name"] == "total_execution_time_ms") {
                execution_time = item["value"].get<float>();
                break;
            }
        }
    }

    // Extract features (mimicking Python's extract_features_from_file)
    std::vector<float> features;
    int nodes_count = 0, edges_count = 0, scheduling_count = 0;
    std::map<std::string, int> op_counts;

    if (data.contains("programming_details")) {
        auto prog_details = data["programming_details"];
        // Nodes
        if (prog_details.contains("Nodes")) {
            nodes_count = prog_details["Nodes"].size();
            for (const auto& node : prog_details["Nodes"]) {
                if (node.contains("Details") && node["Details"].contains("Op histogram")) {
                    for (const auto& op_line : node["Details"]["Op histogram"]) {
                        std::string line = op_line.get<std::string>();
                        size_t pos = line.find(':');
                        if (pos != std::string::npos) {
                            std::string op_name = "op_" + line.substr(0, pos);
                            int count = std::stoi(line.substr(pos + 1));
                            op_counts[op_name] += count;
                        }
                    }
                }
            }
        }
        // Edges
        if (prog_details.contains("Edges")) {
            edges_count = prog_details["Edges"].size();
        }
    }

    // Scheduling data
    if (data.contains("scheduling_data")) {
        scheduling_count = data["scheduling_data"].size();
        auto sched = data["scheduling_data"][0];
        if (sched.contains("Details") && sched["Details"].contains("scheduling_feature")) {
            auto sf = sched["Details"]["scheduling_feature"];
            features.push_back(sf.value("bytes_at_production", 0.0f));
            features.push_back(sf.value("bytes_at_realization", 0.0f));
            features.push_back(sf.value("bytes_at_root", 0.0f));
            features.push_back(sf.value("bytes_at_task", 0.0f));
            features.push_back(sf.value("inner_parallelism", 0.0f));
            features.push_back(sf.value("outer_parallelism", 0.0f));
            features.push_back(sf.value("num_productions", 0.0f));
            features.push_back(sf.value("num_realizations", 0.0f));
            features.push_back(sf.value("num_scalars", 0.0f));
            features.push_back(sf.value("num_vectors", 0.0f));
            features.push_back(sf.value("points_computed_total", 0.0f));
            features.push_back(sf.value("working_set", 0.0f));
        }
    }

    // Basic features
    features.push_back(static_cast<float>(nodes_count));
    features.push_back(static_cast<float>(edges_count));
    features.push_back(static_cast<float>(scheduling_count));
    features.push_back(nodes_count > 0 && edges_count > 0 ? static_cast<float>(nodes_count) / edges_count : 0.0f);

    // Operation counts
    for (const auto& [op, count] : op_counts) {
        features.push_back(static_cast<float>(count));
    }

    // Additional features
    float total_bytes = features[0]; // bytes_at_production
    float total_vectors = features[9]; // num_vectors
    features.push_back(total_bytes);
    features.push_back(total_vectors);
    features.push_back(features[4] * features[5]); // total_parallelism
    features.push_back(total_vectors > 0 ? total_bytes / total_vectors : 0.0f); // bytes_per_vector
    features.push_back(features[11] / (total_bytes > 0 ? total_bytes : 1e-8f)); // memory_pressure

    float total_ops = 0;
    for (size_t i = 16; i < features.size() - 5; ++i) total_ops += features[i];
    features.push_back(nodes_count > 0 ? total_ops / nodes_count : 0.0f); // avg_ops_per_node
    features.push_back(nodes_count > 0 ? static_cast<float>(op_counts.size()) / nodes_count : 0.0f); // op_diversity

    std::cout << "Extracted " << features.size() << " features from " << file_path << std::endl;
    std::cout << "Actual execution time: " << execution_time << " ms" << std::endl;

    return features;
}

int main() {
    try {
        // Load the TorchScript model
        torch::jit::script::Module module = torch::jit::load("lstm_model.pt");
        module.eval();
        std::cout << "Model loaded successfully" << std::endl;

        // Specify the file to test
        std::string file_path = "0_0.json"; // Adjust based on your subfolder structure
        if (!fs::exists(file_path)) {
            throw std::runtime_error("File not found: " + file_path);
        }

        // Extract features
        std::vector<float> features = extract_features_from_json(file_path);

        // Convert to tensor (assuming features match model's input_size)
        torch::Tensor input = torch::from_blob(features.data(), {1, 1, static_cast<long>(features.size())}).to(torch::kFloat32);

        // Move to CUDA if available
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        module.to(device);
        input = input.to(device);

        // Perform inference
        std::vector<torch::jit::IValue> inputs = {input};
        torch::Tensor output = module.forward(inputs).toTensor();

        // Get prediction
        float pred_scaled = output.item<float>();
        std::cout << "Predicted execution time (scaled): " << pred_scaled << std::endl;

        // Note: To get actual ms, you'd need to apply y_scaler's inverse transform
        // This requires y_scaler mean and std, which could be hardcoded or loaded separately

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
