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
    if (data.contains("scheduling_data")) {
        for (const auto& item : data["scheduling_data"]) {
            if (item.contains("name") && item["name"] == "total_execution_time_ms") {
                features["execution_time"] = item["value"].get<float>();
                break;
            }
        }
        if (!features.count("execution_time")) {
            if (!data["scheduling_data"].empty()) {
                features["execution_time"] = data["scheduling_data"].back()["value"].get<float>();
            } else {
                features["execution_time"] = 0.0f;
            }
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
                            // Convert to lowercase
                            for (auto& c : op_name) c = std::tolower(c);
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
            
            // Calculate memory pressure
            if (features["scheduling_count"] > 0) {
                if (total_bytes > 0) {
                    float working_set = features["sched_working_set"];
                    features["memory_pressure"] = working_set / total_bytes;
                } else {
                    features["memory_pressure"] = 0.0f;
                }
            }
        }
    }

    // Calculate avg_ops_per_node and op_diversity if we have nodes
    if (features["nodes_count"] > 0) {
        int total_ops = 0;
        int op_types = 0;
        
        for (const auto& [key, value] : features) {
            if (key.substr(0, 3) == "op_") {
                total_ops += static_cast<int>(value);
                op_types++;
            }
        }
        
        features["avg_ops_per_node"] = total_ops / features["nodes_count"];
        features["op_diversity"] = op_types / features["nodes_count"];
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

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_json_file>\n";
        return -1;
    }

    try {
        // 1. Always use CPU for compatibility
        torch::Device device(torch::kCPU);
        std::cout << "Using CPU device.\n";
        
        // 2. Load model and move to device
        std::cout << "Loading model...\n";
        torch::jit::script::Module model;
        try {
            model = torch::jit::load("lstm_model.pt");
            model.to(device);
            std::cout << "Model loaded successfully.\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            return -1;
        }

        // 3. Process input data
        std::string input_file = argv[1];
        std::cout << "Processing input file: " << input_file << std::endl;
        json data = load_json(input_file);
        auto raw_features = extract_features(data);
        std::cout << "Extracted " << raw_features.size() << " features.\n";

        // 4. Load scaler parameters
        std::cout << "Loading scaler parameters...\n";
        auto scaler_X = load_scaler_params("scaler_X.json");
        auto y_scaler = load_y_scaler_params("scaler_y.json");

        // 5. Scale features
        auto scaled_features = scale_features(raw_features, scaler_X);
        std::cout << "Scaled " << scaled_features.size() << " features.\n";

        // 6. Create input tensor
        torch::Tensor input_tensor = torch::from_blob(
            scaled_features.data(),
            {1, 1, static_cast<int64_t>(scaled_features.size())},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        std::cout << "Input tensor shape: [" 
                  << input_tensor.size(0) << ", " 
                  << input_tensor.size(1) << ", " 
                  << input_tensor.size(2) << "]\n";

        // 7. Run inference
        std::cout << "Running inference...\n";
        torch::NoGradGuard no_grad;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        auto output = model.forward(inputs).toTensor().cpu();
        
        // 8. Post-process the prediction
        float scaled_prediction = output[0][0].item<float>();
        float prediction = inverse_transform_prediction(scaled_prediction, y_scaler);
        
        // 9. Print results
        std::cout << "\nPrediction Results:\n";
        std::cout << "Scaled prediction: " << scaled_prediction << std::endl;
        std::cout << "Predicted execution time: " << prediction << " ms\n";
        
        // 10. Compare to actual if available
        if (raw_features.count("execution_time")) {
            float actual = raw_features["execution_time"];
            float error_pct = std::abs(prediction - actual) / actual * 100;
            std::cout << "Actual execution time: " << actual << " ms\n";
            std::cout << "Error: " << error_pct << "%\n";
        }
        
        // 11. Save results to output JSON
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
