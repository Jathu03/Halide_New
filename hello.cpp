#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <torch/script.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Function to load the schedule representation from JSON
std::vector<float> load_schedule_representation(const std::string& file_path) {
    std::ifstream input_file(file_path);
    json data;
    input_file >> data;
    
    std::vector<float> representation = data["representation"].get<std::vector<float>>();
    return representation;
}

// Function to load target scaler parameters
std::tuple<float, float, bool> load_scaler_y(const std::string& file_path) {
    std::ifstream scaler_file(file_path);
    json scaler_data;
    scaler_file >> scaler_data;
    
    float mean = scaler_data["mean"];
    float scale = scaler_data["scale"];
    bool is_log_transformed = scaler_data["is_log_transformed"];
    
    return std::make_tuple(mean, scale, is_log_transformed);
}

// Function to perform inference
float predict_execution_time(torch::jit::script::Module& model, 
                           const std::vector<float>& representation,
                           float y_mean, float y_scale, bool is_log_transformed) {
    // Convert representation to tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input_tensor = torch::from_blob((void*)representation.data(), 
                                               {1, static_cast<int64_t>(representation.size())}, 
                                               options);
    
    // Reshape for LSTM: [batch_size, sequence_length, input_size]
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0);
    
    // Create input vector
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    // Run inference
    torch::Tensor output = model.forward(inputs).toTensor();
    
    // Inverse transform the prediction
    float prediction = output.item<float>() * y_scale + y_mean;
    
    if (is_log_transformed) {
        prediction = exp(prediction) - 1;  // expm1 equivalent
    }
    
    return prediction;
}

int main() {
    try {
        // Load the trained model
        torch::jit::script::Module model = torch::jit::load("lstm_model.pt");
        model.eval();
        
        // Load the schedule representation
        std::vector<float> representation = load_schedule_representation("schedule_representation.json");
        
        // Load target scaler parameters
        auto [y_mean, y_scale, is_log_transformed] = load_scaler_y("scaler_y.json");
        
        // Perform prediction
        float predicted_time = predict_execution_time(model, representation, y_mean, y_scale, is_log_transformed);
        
        std::cout << "Predicted execution time: " << predicted_time << " ms" << std::endl;
        
        // Print original features for reference
        std::ifstream representation_file("schedule_representation.json");
        json representation_data;
        representation_file >> representation_data;
        
        std::cout << "\nOriginal Features:" << std::endl;
        for (auto& [key, value] : representation_data["original_features"].items()) {
            if (value.is_number()) {
                std::cout << key << ": " << value.get<float>() << std::endl;
            } else if (value.is_string()) {
                std::cout << key << ": " << value.get<std::string>() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
