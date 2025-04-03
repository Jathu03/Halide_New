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
    if (data.contains("scheduling_data")) {
        for (const auto& item : data["scheduling_data"]) {
            if (item.contains("name") && item["name"] == "total_execution_time_ms") {
                features["execution_time"] = item["value"].get<float>();
                break;
            }
        }
        if (!features.count("execution_time")) {
            features["execution_time"] = data["scheduling_data"].back()["value"].get<float>();
        }
    }

    // Initialize counts
    features["nodes_count"] = 0;
    features["edges_count"] = 0;
    features["scheduling_count"] = 0;
    features["node_edge_ratio"] = 0;

    // Process programming_details if exists
    if (data.contains("programming_details")) {
        const auto& pd = data["programming_details"];

        // Count nodes and extract op histograms
        if (pd.contains("Nodes")) {
            features["nodes_count"] = pd["Nodes"].size();
            std::map<std::string, int> op_counts;

            for (const auto& node : pd["Nodes"]) {
                if (node.contains("Details") && node["Details"].contains("Op histogram")) {
                    for (const auto& op_line : node["Details"]["Op histogram"]) {
                        std::string line = op_line.get<std::string>();
                        size_t colon_pos = line.find(':');
                        if (colon_pos != std::string::npos) {
                            std::string op_name = "op_" + line.substr(0, colon_pos);
                            int count = std::stoi(line.substr(colon_pos + 1));
                            op_counts[op_name] += count;
                        }
                    }
                }
            }

            // Add op counts to features
            for (const auto& [op_name, count] : op_counts) {
                features[op_name] = count;
            }
        }

        // Count edges
        if (pd.contains("Edges")) {
            features["edges_count"] = pd["Edges"].size();
        }

        // Calculate node-edge ratio
        if (features["nodes_count"] > 0 && features["edges_count"] > 0) {
            features["node_edge_ratio"] = features["nodes_count"] / features["edges_count"];
        }

        // Process scheduling features
        std::vector<std::string> important_metrics = {
            "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
            "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
            "num_scalars", "num_vectors", "points_computed_total", "working_set"
        };

        const json* scheduling_data = nullptr;
        if (data.contains("scheduling_data")) {
            scheduling_data = &data["scheduling_data"];
        } else if (pd.contains("Schedules")) {
            scheduling_data = &pd["Schedules"];
        }

        if (scheduling_data != nullptr) {
            features["scheduling_count"] = scheduling_data->size();
            
            // Initialize all important metrics to 0
            for (const auto& metric : important_metrics) {
                features["sched_" + metric] = 0;
            }

            float total_bytes = 0;
            float total_vectors = 0;
            float total_parallelism = 0;

            for (const auto& sched : *scheduling_data) {
                if (sched.contains("Details") && sched["Details"].contains("scheduling_feature")) {
                    const auto& sf = sched["Details"]["scheduling_feature"];
                    for (const auto& metric : important_metrics) {
                        if (sf.contains(metric)) {
                            features["sched_" + metric] += sf[metric].get<float>();
                        }
                    }

                    // Sum totals for derived features
                    if (sf.contains("bytes_at_production")) {
                        total_bytes += sf["bytes_at_production"].get<float>();
                    }
                    if (sf.contains("num_vectors")) {
                        total_vectors += sf["num_vectors"].get<float>();
                    }
                    if (sf.contains("inner_parallelism") && sf.contains("outer_parallelism")) {
                        total_parallelism += sf["inner_parallelism"].get<float>() * sf["outer_parallelism"].get<float>();
                    }
                }
            }

            // Add derived features
            features["total_bytes_at_production"] = total_bytes;
            features["total_vectors"] = total_vectors;
            features["total_parallelism"] = total_parallelism;
            
            if (total_vectors > 0) {
                features["bytes_per_vector"] = total_bytes / total_vectors;
            } else {
                features["bytes_per_vector"] = 0;
            }
        }
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

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_json_file>\n";
        return -1;
    }

    try {
        // 1. Determine device (try CUDA first, fallback to CPU)
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Using GPU.\n";
            device = torch::Device(torch::kCUDA);
        } else {
            std::cout << "Using CPU.\n";
        }

        // 2. Load and parse input JSON
        std::string input_file = argv[1];
        json data = load_json(input_file);
        
        // 3. Extract features
        auto raw_features = extract_features(data);
        
        // 4. Load feature scaler parameters
        auto scaler_X = load_scaler_params("scaler_X.json");
        
        // 5. Scale the features
        auto scaled_features = scale_features(raw_features, scaler_X);
        
        // 6. Create input tensor on the correct device
        torch::Tensor input_tensor = torch::from_blob(
            scaled_features.data(),
            {1, 1, static_cast<int64_t>(scaled_features.size())},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);
        
        // 7. Load the model
        torch::jit::script::Module model;
        try {
            // Load model (will be on CPU initially)
            model = torch::jit::load("lstm_model.pt");
            
            // Move model to the same device as input
            model.to(device);
            
            // Verify device alignment
            if (model.parameters().begin()->device() != input_tensor.device()) {
                throw std::runtime_error("Model and input tensor are on different devices!");
            }
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            return -1;
        }
        
        // 8. Run inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        auto output = model.forward(inputs).toTensor().cpu();  // Move output to CPU for processing
        
        // 9. Load target scaler parameters
        json y_scaler = load_json("scaler_y.json");
        float y_mean = y_scaler["mean"].get<float>();
        float y_scale = y_scaler["scale"].get<float>();
        bool is_log_transformed = y_scaler["is_log_transformed"].get<bool>();
        
        // 10. Inverse transform the prediction
        float prediction = output.item<float>() * y_scale + y_mean;
        if (is_log_transformed) {
            prediction = std::expm1(prediction);
        }
        
        // 11. Print results
        std::cout << "\nPrediction Results:\n";
        std::cout << "------------------\n";
        std::cout << "Input file: " << input_file << "\n";
        std::cout << "Device used: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
        std::cout << "Raw execution time: " << raw_features["execution_time"] << " ms\n";
        std::cout << "Predicted execution time: " << prediction << " ms\n";
        std::cout << "Difference: " << std::abs(raw_features["execution_time"] - prediction) << " ms\n";
        std::cout << "Percentage error: " 
                  << (std::abs(raw_features["execution_time"] - prediction) / raw_features["execution_time"]) * 100 
                  << "%\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
