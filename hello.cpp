#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

using json = nlohmann::json;

// Function to extract features from a JSON file (assumed from your setup)
std::map<std::string, float> extract_features_from_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return {};
    }
    json j;
    file >> j;

    std::map<std::string, float> features;
    // Example feature extraction; adjust based on your JSON structure
    features["nodes_count"] = j.value("nodes_count", 0.0f);
    features["edges_count"] = j.value("edges_count", 0.0f);
    // Add more features as per your data
    return features;
}

// Convert features to a tensor for LSTM input
torch::Tensor features_to_tensor(const std::map<std::string, float>& features) {
    // Define the order of features (adjust this list to match your model’s input)
    std::vector<std::string> feature_order = {
        "nodes_count", "edges_count" // Add all features your model expects
    };

    std::vector<float> feature_vec;
    for (const auto& key : feature_order) {
        feature_vec.push_back(features.count(key) ? features.at(key) : 0.0f);
    }

    // Normalize features (replace with your actual mean and std values)
    float mean = 5.0f, std = 2.0f;
    for (auto& val : feature_vec) {
        val = (val - mean) / std;
    }

    // Reshape to {batch_size, sequence_length, input_size}, e.g., {1, 1, feature_count}
    return torch::from_blob(feature_vec.data(), {1, 1, static_cast<long>(feature_vec.size())});
}

int main() {
    // Load the model explicitly on CPU
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("/home/kowrisaan/jathu/Halide_New/lstm_model.pt", torch::kCPU);
        model.eval(); // Set model to evaluation mode
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Example input file (adjust path as needed)
    std::string file_path = "/home/kowrisaan/jathu/Halide_New/synthetic_data/program_50001/0_0.json";
    std::cout << "Processing file: " << file_path << std::endl;

    // Extract features
    auto features = extract_features_from_file(file_path);
    if (features.empty()) {
        std::cerr << "Failed to extract features from " << file_path << std::endl;
        return -1;
    }

    // Convert features to tensor
    torch::Tensor input = features_to_tensor(features);

    // Run inference
    std::vector<torch::jit::IValue> inputs = {input};
    torch::Tensor output;
    try {
        output = model.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }

    // Denormalize output (replace with your actual y_mean and y_std)
    float y_mean = 0.0f, y_std = 1.0f;
    float predicted_time_scaled = output.item<float>();
    float predicted_time = predicted_time_scaled * y_std + y_mean;
    if (predicted_time < 0) {
        predicted_time = std::exp(predicted_time) - 1; // Handle negative predictions if applicable
    }
    std::cout << "Predicted execution time for " << file_path << ": " << predicted_time << " ms" << std::endl;

    return 0;
}
