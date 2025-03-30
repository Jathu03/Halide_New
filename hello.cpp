#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using json = nlohmann::json;

// [get_execution_time and extract_features_from_file unchanged; omitted for brevity]

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

    // Move tensor to CUDA if available
    return torch::from_blob(feature_vec.data(), {1, 1, static_cast<long>(feature_vec.size())}).to(torch::kCUDA);
}

int main() {
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "CUDA not available. Using CPU." << std::endl;
    }

    torch::jit::script::Module model;
    try {
        model = torch::jit::load("/home/kowrisaan/jathu/Halide_New/lstm_model.pt");
        model.eval();
        if (torch::cuda::is_available()) {
            model.to(torch::kCUDA); // Move model to CUDA
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    std::string file_path = "/home/kowrisaan/jathu/Halide_New/synthetic_data/program_50001/0_0.json";
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
