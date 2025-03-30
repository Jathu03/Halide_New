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

// Function to check if CUDA is available (manual implementation)
bool is_cuda_available() {
    // Since torch::cuda::is_available() doesn't work in your environment,
    // we'll return a hardcoded value based on your setup
#ifdef USE_CUDA
    return true;
#else
    return false;
#endif
}

// Function to load the model with proper device handling
torch::jit::script::Module load_model(const std::string& model_path, bool use_cuda) {
    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file
        model = torch::jit::load(model_path);
        
        // Set model to evaluation mode
        model.eval();
        
        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "Model expecting inputs on " << (use_cuda ? "CUDA" : "CPU") << std::endl;
        
        return model;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

// Function to perform inference with device consistency
float predict_execution_time(torch::jit::script::Module& model, 
                           const std::vector<float>& representation,
                           float y_mean, float y_scale, bool is_log_transformed,
                           bool use_cuda) {
    try {
        // Set device based on the use_cuda flag
        torch::Device device = use_cuda ? torch::kCUDA : torch::kCPU;
        
        // The error suggests there's a shape mismatch
        // Let's reshape the tensor properly based on the error message
        // The model expects an input with size 19968
        
        if (representation.size() != 19968) {
            std::cout << "Warning: Input size (" << representation.size() 
                      << ") doesn't match expected size (19968)" << std::endl;
        }
        
        // Create input tensor from the representation vector with the correct sequence length and batch size
        // For LSTM models, the expected shape is [sequence_length, batch_size, input_size]
        // Based on the error, we need to reshape to match the model's expectation
        int seq_length = 1;  // Assuming single step prediction
        int batch_size = 1;  // Single batch
        int input_size = representation.size();  // Feature dimension
        
        // First create a CPU tensor
        torch::Tensor cpu_tensor = torch::from_blob(
            (void*)representation.data(), 
            {input_size}, 
            torch::kFloat32
        ).clone();
        
        // Reshape and move to the correct device
        // We're using the shape [batch_size, input_size] which is common for single time step
        torch::Tensor input_tensor = cpu_tensor.to(device).reshape({batch_size, input_size});
        
        // Print tensor shape for debugging
        std::cout << "Input tensor shape: " << input_tensor.sizes() << std::endl;
        
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
        std::cerr << "Exception details: " << e.what() << std::endl;
        throw std::runtime_error("Error during prediction: " + std::string(e.what()));
    }
}

int main() {
    try {
        // Determine if we should use CUDA
        bool use_cuda = true;  // Set to true because your model expects CUDA tensors
        
        std::cout << "CUDA enabled: " << (use_cuda ? "YES" : "NO") << std::endl;
        
        if (use_cuda && !is_cuda_available()) {
            std::cerr << "WARNING: Model requires CUDA but CUDA is not available on this system." << std::endl;
            std::cerr << "         This will likely cause errors during inference." << std::endl;
        }
        
        // Load model
        std::cout << "Loading model..." << std::endl;
        auto model = load_model("lstm_model.pt", use_cuda);
        
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
        float predicted_time = predict_execution_time(model, representation, y_mean, y_scale, is_log, use_cuda);
        std::cout << "\nPredicted execution time: " << predicted_time << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
