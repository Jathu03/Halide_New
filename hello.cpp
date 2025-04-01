#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <cmath>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Hardcode expected input size from Python training (replace with actual value)
const int EXPECTED_INPUT_SIZE = 47; // Update this based on Python's "Input feature dimension: X"

std::vector<float> extract_features_from_json(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    json data;
    file >> data;

    float execution_time = 0.0;
    if (data.contains("scheduling_data") && data["scheduling_data"].is_array()) {
        for (const auto& item : data["scheduling_data"]) {
            if (item.is_object() && item.contains("name") && item["name"] == "total_execution_time_ms" && item.contains("value")) {
                execution_time = item["value"].get<float>();
                break;
            }
        }
    }

    std::vector<float> features;
    int nodes_count = 0, edges_count = 0, scheduling_count = 0;
    std::map<std::string, int> op_counts;

    if (data.contains("programming_details") && data["programming_details"].is_object()) {
        auto prog_details = data["programming_details"];
        if (prog_details.contains("Nodes") && prog_details["Nodes"].is_array()) {
            nodes_count = prog_details["Nodes"].size();
            for (const auto& node : prog_details["Nodes"]) {
                if (node.contains("Details") && node["Details"].contains("Op histogram") && node["Details"]["Op histogram"].is_array()) {
                    for (const auto& op_line : node["Details"]["Op histogram"]) {
                        if (op_line.is_string()) {
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
        }
        if (prog_details.contains("Edges") && prog_details["Edges"].is_array()) {
            edges_count = prog_details["Edges"].size();
        }
    }

    if (data.contains("scheduling_data") && data["scheduling_data"].is_array()) {
        scheduling_count = data["scheduling_data"].size();
        if (!data["scheduling_data"].empty()) {
            auto sched = data["scheduling_data"][0];
            if (sched.contains("Details") && sched["Details"].is_object() && sched["Details"].contains("scheduling_feature") && sched["Details"]["scheduling_feature"].is_object()) {
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
    }

    features.push_back(static_cast<float>(nodes_count));
    features.push_back(static_cast<float>(edges_count));
    features.push_back(static_cast<float>(scheduling_count));
    features.push_back(nodes_count > 0 && edges_count > 0 ? static_cast<float>(nodes_count) / edges_count : 0.0f);
    for (const auto& [op, count] : op_counts) {
        features.push_back(static_cast<float>(count));
    }

    float total_bytes = features.size() > 0 ? features[0] : 0.0f;
    float total_vectors = features.size() > 9 ? features[9] : 0.0f;
    features.push_back(total_bytes);
    features.push_back(total_vectors);
    features.push_back(features.size() > 5 ? features[4] * features[5] : 0.0f);
    features.push_back(total_vectors > 0 ? total_bytes / total_vectors : 0.0f);
    features.push_back(features.size() > 11 ? features[11] / (total_bytes > 0 ? total_bytes : 1e-8f) : 0.0f);

    float total_ops = 0;
    for (size_t i = 16; i < features.size() - 5; ++i) total_ops += features[i];
    features.push_back(nodes_count > 0 ? total_ops / nodes_count : 0.0f);
    features.push_back(nodes_count > 0 ? static_cast<float>(op_counts.size()) / nodes_count : 0.0f);

    std::cout << "Extracted " << features.size() << " features from " << file_path << std::endl;
    std::cout << "Actual execution time: " << execution_time << " ms" << std::endl;

    // Ensure feature count matches expected input size
    if (features.size() != EXPECTED_INPUT_SIZE) {
        throw std::runtime_error("Feature count (" + std::to_string(features.size()) + 
                                 ") does not match expected input size (" + 
                                 std::to_string(EXPECTED_INPUT_SIZE) + ")");
    }

    return features;
}

int main() {
    try {
        // Load the TorchScript model
        torch::jit::script::Module module = torch::jit::load("lstm_model.pt");
        module.eval();
        std::cout << "Model loaded successfully" << std::endl;

        // Specify the file to test
        std::string file_path = "0_0.json";
        if (!fs::exists(file_path)) {
            throw std::runtime_error("File not found: " + file_path);
        }

        // Extract features
        std::vector<float> features = extract_features_from_json(file_path);

        // Convert to tensor
        torch::Tensor input = torch::from_blob(features.data(), {1, 1, EXPECTED_INPUT_SIZE}).to(torch::kFloat32);

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

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
