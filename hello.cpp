#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>

using json = nlohmann::json;

struct ScalerParams {
    std::vector<std::string> feature_names;
    std::vector<float> means;
    std::vector<float> scales;
};

struct YScalerParams {
    float mean;
    float scale;
    bool is_log_transformed;
};

json load_json(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }
    json data;
    file >> data;
    return data;
}

std::map<std::string, float> extract_features(const json& data) {
    std::map<std::string, float> features;
    if (data.contains("execution_time")) features["execution_time"] = data["execution_time"];
    if (data.contains("nodes_count")) features["nodes_count"] = data["nodes_count"];
    if (data.contains("edges_count")) features["edges_count"] = data["edges_count"];
    return features; // Add more feature extraction as needed
}

ScalerParams load_scaler_params(const std::string& scaler_path) {
    json scaler_data = load_json(scaler_path);
    ScalerParams params;
    params.feature_names = scaler_data["feature_names"].get<std::vector<std::string>>();
    params.means = scaler_data["means"].get<std::vector<float>>();
    params.scales = scaler_data["scales"].get<std::vector<float>>();
    return params;
}

YScalerParams load_y_scaler_params(const std::string& scaler_path) {
    json scaler_data = load_json(scaler_path);
    YScalerParams params;
    params.mean = scaler_data["mean"].get<float>();
    params.scale = scaler_data["scale"].get<float>();
    params.is_log_transformed = scaler_data["is_log_transformed"].get<bool>();
    return params;
}

std::vector<float> scale_features(
    const std::map<std::string, float>& raw_features,
    const ScalerParams& scaler_params,
    torch::Device device
) {
    // Create feature vector in the order of scaler_params.feature_names
    std::vector<float> features_vec;
    for (const auto& name : scaler_params.feature_names) {
        features_vec.push_back(raw_features.count(name) ? raw_features.at(name) : 0.0f);
    }

    // Create tensors on the specified device (e.g., CUDA)
    torch::Tensor features_tensor = torch::tensor(features_vec, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor means_tensor = torch::tensor(scaler_params.means, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor scales_tensor = torch::tensor(scaler_params.scales, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Perform scaling on the device
    torch::Tensor scaled_tensor = (features_tensor - means_tensor) / scales_tensor;

    // Move back to CPU and convert to vector
    auto data_ptr = scaled_tensor.cpu().contiguous().data_ptr<float>();
    std::vector<float> scaled_features(data_ptr, data_ptr + scaled_tensor.numel());

    return scaled_features;
}

float inverse_transform_prediction(float scaled_prediction, const YScalerParams& y_scaler, torch::Device device) {
    torch::Tensor scaled_pred_tensor = torch::tensor({scaled_prediction}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor mean_tensor = torch::tensor({y_scaler.mean}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor scale_tensor = torch::tensor({y_scaler.scale}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor unscaled_tensor = scaled_pred_tensor * scale_tensor + mean_tensor;
    float unscaled = unscaled_tensor.item<float>();
    return y_scaler.is_log_transformed ? std::exp(unscaled) - 1.0f : unscaled;
}

float run_prediction(const std::vector<float>& scaled_features, const std::string& model_path, torch::Device device) {
    torch::Tensor input_tensor = torch::from_blob(
        const_cast<float*>(scaled_features.data()),
        {1, 1, static_cast<int64_t>(scaled_features.size())},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    ).clone();

    torch::jit::script::Module model = torch::jit::load(model_path, device);
    model.eval();

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {input_tensor};
    auto output = model.forward(inputs).toTensor();
    return output[0][0].item<float>();
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <json_file.json>\n";
        return -1;
    }

    // Set device to CUDA if available
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    try {
        std::string model_path = "lstm_model.pt";
        std::string input_file = argv[1];
        json data = load_json(input_file);
        auto raw_features = extract_features(data);

        auto scaler_X = load_scaler_params("scaler_X.json");
        auto y_scaler = load_y_scaler_params("scaler_y.json");

        auto scaled_features = scale_features(raw_features, scaler_X, device);
        float scaled_prediction = run_prediction(scaled_features, model_path, device);
        float prediction = inverse_transform_prediction(scaled_prediction, y_scaler, device);

        std::cout << "Predicted execution time: " << prediction << " ms\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
