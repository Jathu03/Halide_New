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

float get_execution_time(const json& data) {
    if (!data.contains("scheduling_data")) {
        throw std::runtime_error("'scheduling_data' not found in JSON");
    }

    for (const auto& item : data["scheduling_data"]) {
        if (item.contains("name") && item["name"] == "total_execution_time_ms") {
            return item["value"].get<float>();
        }
    }
    return data["scheduling_data"].back()["value"].get<float>();
}

std::map<std::string, float> extract_features(const json& data) {
    std::map<std::string, float> features;
    features["execution_time"] = get_execution_time(data);
    features["nodes_count"] = 0;
    features["edges_count"] = 0;
    features["scheduling_count"] = 0;
    features["node_edge_ratio"] = 0;

    if (data.contains("programming_details")) {
        const auto& pd = data["programming_details"];

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
            for (const auto& [op_name, count] : op_counts) {
                features[op_name] = count;
            }
        }

        if (pd.contains("Edges")) {
            features["edges_count"] = pd["Edges"].size();
        }

        if (features["nodes_count"] > 0 && features["edges_count"] > 0) {
            features["node_edge_ratio"] = features["nodes_count"] / features["edges_count"];
        }

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

            features["total_bytes_at_production"] = total_bytes;
            features["total_vectors"] = total_vectors;
            features["total_parallelism"] = total_parallelism;
            
            features["bytes_per_vector"] = total_vectors > 0 ? total_bytes / total_vectors : 0;
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

int main() {
    try {
        // 1. Load and parse input
        json data = load_json("0_0.json");
        
        // 2. Extract features
        auto raw_features = extract_features(data);
        
        // 3. Load and scale features
        auto scaler_X = load_scaler_params("scaler_X.json");
        auto scaled_features = scale_features(raw_features, scaler_X);
        
        // 4. Create tensor on CPU
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(
            scaled_features.data(),
            {1, 1, static_cast<int64_t>(scaled_features.size())},
            options
        ).clone();
        
        // 5. Load model and move to CPU
        torch::jit::script::Module model;
        try {
            model = torch::jit::load("lstm_model.pt", torch::kCPU);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return -1;
        }
        
        // 6. Run inference
        auto output = model.forward({input_tensor}).toTensor();
        
        // 7. Process output
        json y_scaler = load_json("scaler_y.json");
        float prediction = output.item<float>() * y_scaler["scale"].get<float>() + y_scaler["mean"].get<float>();
        
        if (y_scaler["is_log_transformed"].get<bool>()) {
            prediction = std::expm1(prediction);
        }
        
        std::cout << "Predicted execution time: " << prediction << " ms\n";
        std::cout << "Actual execution time: " << raw_features["execution_time"] << " ms\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
