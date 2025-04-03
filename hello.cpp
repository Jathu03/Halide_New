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

  // Extract execution time
  if (data.contains("execution_time")) {
    features["execution_time"] = data["execution_time"];
  }

  // Extract node and edge counts
  if (data.contains("nodes_count")) {
    features["nodes_count"] = data["nodes_count"];
  }
  if (data.contains("edges_count")) {
    features["edges_count"] = data["edges_count"];
  }
  if (data.contains("node_edge_ratio")) {
    features["node_edge_ratio"] = data["node_edge_ratio"];
  }

  // Extract operation counts (op_* features)
  for (auto& [key, value] : data.items()) {
    if (key.find("op_") == 0) {
      features[key] = value;
    }
  }

  // Extract scheduling features
  if (data.contains("sched_bytes_at_production")) {
    features["sched_bytes_at_production"] = data["sched_bytes_at_production"];
  }
  if (data.contains("sched_bytes_at_realization")) {
    features["sched_bytes_at_realization"] = data["sched_bytes_at_realization"];
  }
  if (data.contains("sched_inner_parallelism")) {
    features["sched_inner_parallelism"] = data["sched_inner_parallelism"];
  }
  if (data.contains("sched_outer_parallelism")) {
    features["sched_outer_parallelism"] = data["sched_outer_parallelism"];
  }
  if (data.contains("sched_num_vectors")) {
    features["sched_num_vectors"] = data["sched_num_vectors"];
  }
  if (data.contains("sched_working_set")) {
    features["sched_working_set"] = data["sched_working_set"];
  }

  // Extract derived features
  if (data.contains("total_bytes_at_production")) {
    features["total_bytes_at_production"] = data["total_bytes_at_production"];
  }
  if (data.contains("total_vectors")) {
    features["total_vectors"] = data["total_vectors"];
  }
  if (data.contains("total_parallelism")) {
    features["total_parallelism"] = data["total_parallelism"];
  }
  if (data.contains("bytes_per_vector")) {
    features["bytes_per_vector"] = data["bytes_per_vector"];
  }
  if (data.contains("memory_pressure")) {
    features["memory_pressure"] = data["memory_pressure"];
  }
  if (data.contains("avg_ops_per_node")) {
    features["avg_ops_per_node"] = data["avg_ops_per_node"];
  }
  if (data.contains("op_diversity")) {
    features["op_diversity"] = data["op_diversity"];
  }

  return features;
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
    const ScalerParams& scaler_params
) {
  std::vector<float> scaled_features(scaler_params.feature_names.size(), 0.0f);

  for (size_t i = 0; i < scaler_params.feature_names.size(); ++i) {
    const std::string& feature_name = scaler_params.feature_names[i];
    float value = raw_features.count(feature_name) ? raw_features.at(feature_name) : 0.0f;
    scaled_features[i] = (value - scaler_params.means[i]) / scaler_params.scales[i];
  }

  return scaled_features;
}

float inverse_transform_prediction(float scaled_prediction, const YScalerParams& y_scaler) {
  float unscaled = scaled_prediction * y_scaler.scale + y_scaler.mean;
  if (y_scaler.is_log_transformed) {
    return std::exp(unscaled) - 1.0f;  // expm1
  }
  return unscaled;
}

float run_prediction(const std::vector<float>& scaled_features, const std::string& model_path) {
  // Check if CUDA is available
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  // Create input tensor and move to the same device as model
  torch::Tensor input_tensor = torch::from_blob(
      const_cast<float*>(scaled_features.data()),
      {1, 1, static_cast<int64_t>(scaled_features.size())},
      torch::TensorOptions().dtype(torch::kFloat32)
  ).clone().to(device);

  std::cout << "Input tensor created with shape: ["
            << input_tensor.size(0) << ", "
            << input_tensor.size(1) << ", "
            << input_tensor.size(2) << "] on "
            << (input_tensor.device().is_cuda() ? "CUDA" : "CPU") << "\n";

  // Load the model
  torch::jit::script::Module model;
  try {
    // Load model and move to device
    model = torch::jit::load(model_path);
    model.to(device);
    model.eval();
    std::cout << "Model loaded successfully on "
              << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model: " << e.what() << std::endl;
    throw;
  }

  // Run inference
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor);

  // Forward pass
  auto output = model.forward(inputs).toTensor();

  // Get prediction value
  float scaled_prediction = output[0][0].item<float>();

  return scaled_prediction;
}
int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <json_file.json>\n";
    return -1;
  }

  try {
    std::string model_path = "lstm_model.pt";

    // Process input data
    std::string input_file = argv[1];
    std::cout << "Processing input file: " << input_file << std::endl;
    json data = load_json(input_file);
    auto raw_features = extract_features(data);
    std::cout << "Extracted " << raw_features.size() << " features.\n";

    // Load scaler parameters
    std::cout << "Loading scaler parameters...\n";
    auto scaler_X = load_scaler_params("scaler_X.json");
    auto y_scaler = load_y_scaler_params("scaler_y.json");

    // Scale features
    auto scaled_features = scale_features(raw_features, scaler_X);
    std::cout << "Scaled " << scaled_features.size() << " features.\n";

    // Run prediction using CPU only
    float scaled_prediction = run_prediction(scaled_features, model_path);
    float prediction = inverse_transform_prediction(scaled_prediction, y_scaler);

    // Print results
    std::cout << "\nPrediction Results:\n";
    std::cout << "Scaled prediction: " << scaled_prediction << std::endl;
    std::cout << "Predicted execution time: " << prediction << " ms\n";

    // Compare to actual if available
    if (raw_features.count("execution_time")) {
      float actual = raw_features["execution_time"];
      float error_pct = std::abs(prediction - actual) / actual * 100;
      std::cout << "Actual execution time: " << actual << " ms\n";
      std::cout << "Error: " << error_pct << "%\n";
    }

    // Save results to output JSON
    json result;
    result["input_file"] = input_file;
    result["predicted_execution_time_ms"] = prediction;
    if (raw_features.count("execution_time")) {
      result["actual_execution_time_ms"] = raw_features["execution_time"];
      result["error_percentage"] = std::abs(prediction - raw_features["execution_time"])
                                     / raw_features["execution_time"] * 100;
    }

    std::string output_file = input_file + ".prediction.json";
    std::ofstream out_file(output_file);
    out_file << result.dump(4);
    std::cout << "Prediction saved to: " << output_file << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
