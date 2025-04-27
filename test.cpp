#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Constants
const std::vector<std::string> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    "sched_num_productions", "sched_points_computed_total", "sched_innermost_loop_extent",
    "sched_inner_parallelism", "sched_outer_parallelism", "sched_bytes_at_realization",
    "sched_bytes_at_production", "sched_bytes_at_root", "sched_unique_bytes_read_per_realization",
    "sched_working_set", "sched_vector_size", "sched_num_vectors", "sched_num_scalars",
    "sched_bytes_at_task", "sched_working_set_at_task", "sched_working_set_at_production",
    "sched_working_set_at_realization", "sched_working_set_at_root", "total_parallelism",
    "scheduling_count", "total_bytes_at_production", "total_vectors", "computation_efficiency",
    "memory_pressure", "memory_utilization_ratio", "bytes_processing_rate", "bytes_per_parallelism",
    "bytes_per_vector", "nodes_count", "edges_count", "node_edge_ratio", "nodes_per_schedule",
    "op_diversity",
    "op_add", "op_sub", "op_mul", "op_div", "op_mod", "op_eq", "op_ne", "op_lt", "op_le",
    "op_or", "op_and", "op_not", "op_min", "op_max", "op_constant", "op_variable",
    "op_funccall", "op_imagecall", "op_externcall", "op_let", "op_param",
    "memory_transpose_0", "memory_transpose_1", "memory_transpose_2", "memory_transpose_3",
    "memory_slice_0", "memory_slice_1", "memory_slice_2", "memory_slice_3",
    "memory_broadcast_0", "memory_broadcast_1", "memory_broadcast_2", "memory_broadcast_3",
    "memory_pointwise_0", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3"
};

// Structure to hold scaler parameters
struct ScalerParams {
    std::vector<double> X_scalar_center;
    std::vector<double> X_scalar_scale;
    std::vector<double> y_center;
    std::vector<double> y_scale;
    std::vector<std::string> feature_columns;
};

// Function to load scaler parameters from JSON file
ScalerParams loadScalerParams(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open scaler params file: " + filename);
    }
    
    json j;
    file >> j;
    
    ScalerParams params;
    params.X_scalar_center = j["X_scalar_center"].get<std::vector<double>>();
    params.X_scalar_scale = j["X_scalar_scale"].get<std::vector<double>>();
    params.y_center = j["y_center"].get<std::vector<double>>();
    params.y_scale = j["y_scale"].get<std::vector<double>>();
    params.feature_columns = j["feature_columns"].get<std::vector<std::string>>();
    
    return params;
}

// Function to extract features from JSON similar to the Python implementation
std::map<std::string, double> extractFeatures(const json& jsonData) {
    std::map<std::string, double> features;
    
    // Initialize all features to 0
    for (const auto& feature : FIXED_FEATURES) {
        features[feature] = 0.0;
    }
    
    // Extract global features
    auto it = std::find_if(jsonData["children"].begin(), jsonData["children"].end(),
                         [](const json& child) { return child["name"] == "Global Features"; });
    if (it != jsonData["children"].end()) {
        const auto& globalNode = *it;
        features["cache_hits"] = globalNode.contains("cache_hits") ? globalNode["cache_hits"].get<double>() : 0.0;
        features["cache_misses"] = globalNode.contains("cache_misses") ? globalNode["cache_misses"].get<double>() : 0.0;
        features["execution_time_ms"] = globalNode.contains("execution_time_ms") ? globalNode["execution_time_ms"].get<double>() : 0.0;
    }
    
    // Extract op_histogram features
    std::map<std::string, int> opHistogram;
    for (const auto& node : jsonData["children"]) {
        if (node.contains("op_histogram")) {
            for (auto& [op, count] : node["op_histogram"].items()) {
                std::string opLower = op;
                std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
                opHistogram[opLower] += count.get<int>();
            }
        }
    }
    
    for (const auto& [op, count] : opHistogram) {
        features["op_" + op] = count;
    }
    
    // Extract memory patterns
    std::map<std::string, std::vector<double>> memoryPatterns;
    for (const auto& node : jsonData["children"]) {
        if (node.contains("memory_patterns")) {
            for (auto& [pattern, values] : node["memory_patterns"].items()) {
                std::string patternLower = pattern;
                std::transform(patternLower.begin(), patternLower.end(), patternLower.begin(), ::tolower);
                
                std::vector<double> currValues;
                if (values.is_array()) {
                    for (const auto& val : values) {
                        currValues.push_back(val.get<double>());
                    }
                }
                
                // Ensure 4 elements, padding with zeros if needed
                while (currValues.size() < 4) {
                    currValues.push_back(0.0);
                }
                
                // If pattern doesn't exist, initialize it
                if (memoryPatterns.find(patternLower) == memoryPatterns.end()) {
                    memoryPatterns[patternLower] = {0.0, 0.0, 0.0, 0.0};
                }
                
                // Add to existing values
                for (size_t i = 0; i < 4; ++i) {
                    memoryPatterns[patternLower][i] += currValues[i];
                }
            }
        }
    }
    
    for (const auto& [pattern, values] : memoryPatterns) {
        for (size_t i = 0; i < values.size(); ++i) {
            features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
        }
    }
    
    // Extract scheduling features
    std::vector<std::string> schedulingKeys = {
        "num_realizations", "num_productions", "points_computed_total", "innermost_loop_extent",
        "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
        "bytes_at_root", "unique_bytes_read_per_realization", "working_set", "vector_size",
        "num_vectors", "num_scalars", "bytes_at_task", "working_set_at_task", "working_set_at_production",
        "working_set_at_realization", "working_set_at_root"
    };
    
    std::map<std::string, double> schedulingSums;
    int nodeCount = 0;
    
    for (const auto& node : jsonData["children"]) {
        if (node.contains("scheduling")) {
            nodeCount++;
            for (const auto& key : schedulingKeys) {
                if (node["scheduling"].contains(key)) {
                    schedulingSums[key] += node["scheduling"][key].get<double>();
                }
            }
        }
    }
    
    for (const auto& key : schedulingKeys) {
        if ((key == "inner_parallelism" || key == "outer_parallelism") && nodeCount > 0) {
            features["sched_" + key] = schedulingSums[key] / nodeCount;
        } else {
            features["sched_" + key] = schedulingSums[key];
        }
    }
    
    // Derived features with division-by-zero protection
    features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"];
    features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"];
    features["total_bytes_at_production"] = features["sched_bytes_at_production"];
    features["total_vectors"] = features["sched_num_vectors"];
    
    features["computation_efficiency"] = features["sched_bytes_at_realization"] != 0 ?
        features["sched_points_computed_total"] / features["sched_bytes_at_realization"] : 0;
    
    features["memory_pressure"] = features["sched_bytes_at_root"] != 0 ?
        features["sched_working_set"] / features["sched_bytes_at_root"] : 0;
    
    features["memory_utilization_ratio"] = features["sched_bytes_at_task"] != 0 ?
        features["sched_unique_bytes_read_per_realization"] / features["sched_bytes_at_task"] : 0;
    
    features["bytes_processing_rate"] = features["execution_time_ms"] != 0 ?
        features["sched_bytes_at_realization"] / features["execution_time_ms"] : 0;
    
    features["bytes_per_parallelism"] = features["total_parallelism"] != 0 ?
        features["sched_bytes_at_task"] / features["total_parallelism"] : 0;
    
    features["bytes_per_vector"] = features["sched_num_vectors"] != 0 ?
        features["sched_bytes_at_realization"] / features["sched_num_vectors"] : 0;
    
    int nodesCount = jsonData["children"].size();
    int edgesCount = 0;
    for (const auto& node : jsonData["children"]) {
        if (node.contains("children")) {
            edgesCount += node["children"].size();
        }
    }
    
    features["nodes_count"] = nodesCount;
    features["edges_count"] = edgesCount;
    features["node_edge_ratio"] = edgesCount != 0 ? static_cast<double>(nodesCount) / edgesCount : nodesCount;
    features["nodes_per_schedule"] = features["scheduling_count"] != 0 ?
        nodesCount / features["scheduling_count"] : nodesCount;
    
    // Count op diversity
    int opDiversity = 0;
    for (const auto& [key, value] : features) {
        if (key.substr(0, 3) == "op_" && value > 0) {
            opDiversity++;
        }
    }
    features["op_diversity"] = opDiversity;
    
    return features;
}

// Transform data similarly to the Python code
std::vector<torch::Tensor> prepareDataForModel(
    const std::map<std::string, double>& features,
    const ScalerParams& scalerParams
) {
    // Create log transformations for specified features
    std::map<std::string, double> transformedFeatures = features;
    
    // Apply log1p transformation to selected features
    std::vector<std::string> skewedFeatures = {"cache_hits", "bytes_processing_rate", "sched_bytes_at_task", "computation_efficiency"};
    for (const auto& feature : skewedFeatures) {
        if (transformedFeatures.find(feature) != transformedFeatures.end()) {
            transformedFeatures["log_" + feature] = std::log1p(transformedFeatures[feature]);
            transformedFeatures.erase(feature);
        }
    }
    
    // Create sequence input tensor (always 3 length sequence in this model)
    const int sequenceLength = 3;
    std::vector<float> sequenceData;
    
    // Add each fixed feature to the sequence data
    for (int i = 0; i < sequenceLength; ++i) {
        for (const auto& feature : FIXED_FEATURES) {
            sequenceData.push_back(features.find(feature) != features.end() ? features.at(feature) : 0.0f);
        }
    }
    
    // Create the 3D sequence tensor [1, sequence_length, feature_count]
    auto sequenceTensor = torch::from_blob(sequenceData.data(), 
                                           {1, sequenceLength, static_cast<int64_t>(FIXED_FEATURES.size())}, 
                                           torch::kFloat32).clone();
    
    // Create scalar input tensor
    std::vector<float> scalarData;
    
    // Add each scalar feature in the correct order according to scalerParams.feature_columns
    for (const auto& featureCol : scalerParams.feature_columns) {
        if (transformedFeatures.find(featureCol) != transformedFeatures.end()) {
            scalarData.push_back(transformedFeatures.at(featureCol));
        } else {
            scalarData.push_back(0.0f);  // Default to 0 if feature not found
        }
    }
    
    // Create the tensor
    auto scalarTensor = torch::from_blob(scalarData.data(), 
                                         {1, static_cast<int64_t>(scalarData.size())}, 
                                         torch::kFloat32).clone();
    
    // Apply robust scaling to scalar tensor
    for (int i = 0; i < scalarData.size(); ++i) {
        float center = i < scalerParams.X_scalar_center.size() ? scalerParams.X_scalar_center[i] : 0.0f;
        float scale = i < scalerParams.X_scalar_scale.size() ? scalerParams.X_scalar_scale[i] : 1.0f;
        scale = scale != 0 ? scale : 1.0f;  // Avoid division by zero
        
        scalarTensor[0][i] = (scalarTensor[0][i].item<float>() - center) / scale;
    }
    
    // Replace NaN values with 0
    scalarTensor = torch::nan_to_num(scalarTensor, 0.0, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
    
    return {sequenceTensor, scalarTensor};
}

// Inverse transform the model's output to get the actual prediction
double inverseTransformPrediction(float scaled_value, const ScalerParams& scalerParams) {
    // Un-scale the value
    double center = scalerParams.y_center[0];
    double scale = scalerParams.y_scale[0];
    double unscaled = scaled_value * scale + center;
    
    // Inverse of log1p is expm1
    return std::expm1(unscaled);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_json_file> [model_path] [scaler_params_path]" << std::endl;
        return 1;
    }
    
    std::string jsonFilePath = argv[1];
    std::string modelPath = (argc > 2) ? argv[2] : "model.pt";
    std::string scalerParamsPath = (argc > 3) ? argv[3] : "scaler_params.json";
    
    try {
        // Load the TorchScript model
        std::cout << "Loading model from " << modelPath << std::endl;
        torch::jit::script::Module model;
        try {
            model = torch::jit::load(modelPath);
            model.eval();
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            return 1;
        }
        
        // Load scaler parameters
        std::cout << "Loading scaler parameters from " << scalerParamsPath << std::endl;
        ScalerParams scalerParams = loadScalerParams(scalerParamsPath);
        
        // Load and process the JSON file
        std::cout << "Processing JSON file: " << jsonFilePath << std::endl;
        std::ifstream file(jsonFilePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open JSON file: " << jsonFilePath << std::endl;
            return 1;
        }
        
        json jsonData;
        file >> jsonData;
        
        // Extract features
        std::map<std::string, double> features = extractFeatures(jsonData);
        
        // Print some key features
        std::cout << "Extracted features:" << std::endl;
        std::cout << "  cache_hits: " << features["cache_hits"] << std::endl;
        std::cout << "  execution_time_ms: " << features["execution_time_ms"] << std::endl;
        std::cout << "  sched_bytes_at_realization: " << features["sched_bytes_at_realization"] << std::endl;
        
        // Prepare tensors for the model
        std::vector<torch::Tensor> modelInputs = prepareDataForModel(features, scalerParams);
        torch::Tensor sequenceInput = modelInputs[0];
        torch::Tensor scalarInput = modelInputs[1];
        
        // Perform inference
        std::cout << "Running inference..." << std::endl;
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(sequenceInput);
        inputs.push_back(scalarInput);
        
        at::Tensor output = model.forward(inputs).toTensor();
        
        // Process the output
        float predictedScaled = output[0][0].item<float>();
        double predictedExecutionTime = inverseTransformPrediction(predictedScaled, scalerParams);
        
        std::cout << "Prediction Results:" << std::endl;
        std::cout << "  Actual execution time from JSON: " << features["execution_time_ms"] << " ms" << std::endl;
        std::cout << "  Predicted execution time: " << predictedExecutionTime << " ms" << std::endl;
        
        double errorPercentage = features["execution_time_ms"] > 0 ?
            std::abs(features["execution_time_ms"] - predictedExecutionTime) / features["execution_time_ms"] * 100 : 0;
        std::cout << "  Error percentage: " << errorPercentage << "%" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
