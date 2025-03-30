#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <torch/script.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Function to load target scaler parameters
std::tuple<float, float, bool> load_scaler_y(const std::string& file_path) {
    std::ifstream scaler_file(file_path);
    if (!scaler_file.is_open()) {
        throw std::runtime_error("Failed to open scaler file: " + file_path);
    }

    json scaler_data;
    try {
        scaler_file >> scaler_data;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + file_path + ": " + e.what());
    }

    return std::make_tuple(
        scaler_data["mean"].get<float>(),
        scaler_data["scale"].get<float>(),
        scaler_data["is_log_transformed"].get<bool>()
    );
}

// Function to load the schedule representation
std::vector<float> load_schedule_representation(const std::string& file_path) {
    std::ifstream input_file(file_path);
    if (!input_file.is_open()) {
        throw std::runtime_error("Failed to open representation file: " + file_path);
    }

    json data;
    try {
        input_file >> data;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + file_path + ": " + e.what());
    }

    if (!data.contains("representation")) {
        throw std::runtime_error("Missing 'representation' in JSON file");
    }

    return data["representation"].get<std::vector<float>>();
}

// Function to load the model with proper device handling
torch::jit::script::Module load_model(const std::string& model_path, bool use_cuda) {
    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file
        model = torch::jit::load(model_path);

        // Move to appropriate device
        if (use_cuda) {
            model.to(torch::kCUDA);
            std::cout << "Model moved to CUDA" << std::endl;
        } else {
            model.to(torch::kCPU);
            std::cout << "Model moved to CPU" << std::endl;
        }

        model.eval();
        return model;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

// Function to perform inference with device consistency
float predict_execution_time(torch::jit::script::Module& model, 
                          const std::vector<float>& representation,
                          float y_mean, float y_scale, bool is_log_transformed) {
    try {
        // Default to CPU device
        torch::Device device = torch::kCPU;
        
        // Just use the device the model was moved to
        // We know this from the use_cuda parameter that was passed to load_model
        
        // Convert representation to tensor and move to correct device
        torch::Tensor input_tensor = torch::from_blob(
            (void*)representation.data(), 
            {1, static_cast<int64_t>(representation.size())}, 
            torch::kFloat32
        ).clone().unsqueeze(0).unsqueeze(0).to(device);

        // Create input vector
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Run inference
        torch::Tensor output = model.forward(inputs).toTensor();

        // Inverse transform the prediction
        float prediction = output.item<float>() * y_scale + y_mean;

        if (is_log_transformed) {
            prediction = exp(prediction) - 1.0f;
        }

        return prediction;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error during prediction: " + std::string(e.what()));
    }
}

int main() {
    try {
        // Set CUDA flag based on your build configuration
        bool use_cuda = false;  // Set to true if you want to use CUDA and know it's available
        
        std::cout << "CUDA enabled: " << (use_cuda ? "YES" : "NO") << std::endl;

        // Load model
        std::cout << "Loading model..." << std::endl;
        auto model = load_model("lstm_model.pt", use_cuda);
        std::cout << "Model loaded successfully" << std::endl;

        // Load representation
        std::cout << "Loading representation..." << std::endl;
        auto representation = load_schedule_representation("schedule_representation.json");
        std::cout << "Representation size: " << representation.size() << std::endl;

        // Load scaler parameters
        std::cout << "Loading scaler parameters..." << std::endl;
        auto [y_mean, y_scale, is_log] = load_scaler_y("scaler_y.json");
        std::cout << "Scaler parameters loaded (mean=" << y_mean 
                  << ", scale=" << y_scale 
                  << ", log=" << is_log << ")" << std::endl;

        // Perform prediction
        std::cout << "Running prediction..." << std::endl;
        float predicted_time = predict_execution_time(model, representation, y_mean, y_scale, is_log);
        std::cout << "\nPredicted execution time: " << predicted_time << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
