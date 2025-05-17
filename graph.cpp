#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cctype>

// Using nlohmann::json for JSON parsing
using json = nlohmann::json;

// Fixed features as defined in the Python code
const std::vector<std::string> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    "sched_num_productions", "sched_points_computed_total", "sched_inner_parallelism",
    "sched_outer_parallelism", "sched_bytes_at_realization", "sched_bytes_at_production",
    "sched_bytes_at_root", "sched_bytes_at_task", "sched_working_set", "sched_num_vectors",
    "sched_num_scalars", "total_parallelism", "scheduling_count", "total_bytes_at_production",
    "total_vectors", "computation_efficiency", "memory_pressure", "memory_utilization_ratio",
    "bytes_processing_rate", "bytes_per_parallelism", "bytes_per_vector", "nodes_count",
    "edges_count", "node_edge_ratio", "nodes_per_schedule", "op_diversity",
    "op_add", "op_sub", "op_mul", "op_div", "op_mod", "op_eq", "op_ne", "op_lt", "op_le",
    "op_or", "op_and", "op_not", "op_min", "op_max", "op_constant", "op_variable",
    "op_funccall", "op_imagecall", "op_externcall", "op_let", "op_param",
    "memory_transpose_0", "memory_transpose_1", "memory_transpose_2", "memory_transpose_3",
    "memory_slice_0", "memory_slice_1", "memory_slice_2", "memory_slice_3",
    "memory_broadcast_0", "memory_broadcast_1", "memory_broadcast_2", "memory_broadcast_3",
    "memory_pointwise_0", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3"
};

// Utility to convert string to lowercase
std::string to_lowercase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
    return result;
}

// Load JSON file
json load_json(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    json j;
    file >> j;
    return j;
}

// Load scaler parameters
std::pair<std::vector<double>, std::vector<double>> load_scaler_params(const std::string& file_path) {
    json j = load_json(file_path);
    std::vector<double> center = j["center"].get<std::vector<double>>();
    std::vector<double> scale = j["scale"].get<std::vector<double>>();
    if (center.size() != scale.size()) {
        throw std::runtime_error("Mismatch in scaler parameters dimensions in " + file_path);
    }
    return {center, scale};
}

// Load model metadata
json load_model_metadata(const std::string& file_path) {
    return load_json(file_path);
}

// Feature extraction (mirrors Python's extract_features)
std::map<std::string, double> extract_features(const json& json_data) {
    std::map<std::string, double> features;
    try {
        // Validate JSON structure
        if (!json_data.contains("without_extern")) {
            std::cerr << "Missing 'without_extern' key in JSON" << std::endl;
            return {};
        }
        const auto& without_extern = json_data["without_extern"];
        if (!without_extern.contains("global_features")) {
            std::cerr << "Missing 'global_features' key in JSON" << std::endl;
            return {};
        }
        const auto& global_features = without_extern["global_features"];

        // Extract and validate execution time
        if (!global_features.contains("execution_time_ms") || global_features["execution_time_ms"].is_null()) {
            std::cerr << "Missing or null 'execution_time_ms' in global_features" << std::endl;
            return {};
        }
        double execution_time_ms = global_features["execution_time_ms"].get<double>();
        if (execution_time_ms <= 0) {
            std::cerr << "Invalid execution_time_ms: " << execution_time_ms << std::endl;
            return {};
        }

        features["execution_time_ms"] = execution_time_ms;
        features["cache_hits"] = global_features.value("cache_hits", 0.0);
        features["cache_misses"] = global_features.value("cache_misses", 0.0);

        // Extract node and edge counts
        auto nodes = without_extern.value("nodes", json::array());
        auto edges = without_extern.value("edges", json::array());
        features["nodes_count"] = nodes.size();
        features["edges_count"] = edges.size();
        features["node_edge_ratio"] = features["nodes_count"] / (features["edges_count"] + 1e-8);

        // Extract operation counts
        std::map<std::string, double> op_counts;
        std::map<std::string, std::vector<double>> memory_patterns;
        for (const auto& pattern : {"transpose", "slice", "broadcast", "pointwise"}) {
            memory_patterns[pattern] = std::vector<double>(4, 0.0);
        }

        for (const auto& node : nodes) {
            auto stages = node.value("stages", json::array());
            for (const auto& stage : stages) {
                auto pipeline_features = stage.value("pipeline_features", json::object());
                auto op_hist = pipeline_features.value("op_histogram", json::object()).value("Float", json::object());
                for (const auto& [op, count] : op_hist.items()) {
                    op_counts["op_" + to_lowercase(op)] += count.get<double>();
                }

                auto mem_access = pipeline_features.value("memory_access_patterns", json::object()).value("Float", json::object());
                for (const auto& [pattern, values] : mem_access.items()) {
                    auto pattern_lower = to_lowercase(pattern);
                    if (memory_patterns.find(pattern_lower) == memory_patterns.end()) {
                        memory_patterns[pattern_lower] = std::vector<double>(4, 0.0);
                    }
                    auto vals = values.get<std::vector<double>>();
                    for (size_t i = 0; i < std::min(vals.size(), size_t(4)); ++i) {
                        memory_patterns[pattern_lower][i] += vals[i];
                    }
                }
            }
        }

        features.insert(op_counts.begin(), op_counts.end());
        for (const auto& [pattern, values] : memory_patterns) {
            for (size_t i = 0; i < values.size(); ++i) {
                features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
            }
        }

        // Extract scheduling features
        std::vector<json> scheduling_features;
        for (const auto& node : nodes) {
            auto stages = node.value("stages", json::array());
            for (const auto& stage : stages) {
                auto sched = stage.value("schedule_features", json::object());
                scheduling_features.push_back(sched);
            }
        }

        features["scheduling_count"] = scheduling_features.size();

        if (!scheduling_features.empty()) {
            std::vector<std::string> important_metrics = {
                "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
                "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
                "num_scalars", "num_vectors", "points_computed_total", "working_set"
            };

            for (const auto& metric : important_metrics) {
                double sum = 0.0;
                for (const auto& sf : scheduling_features) {
                    sum += sf.value(metric, 0.0);
                }
                features["sched_" + metric] = sum;
            }

            features["total_bytes_at_production"] = features["sched_bytes_at_production"];
            features["total_vectors"] = features["sched_num_vectors"];
            double total_parallelism = 0.0;
            for (const auto& sf : scheduling_features) {
                total_parallelism += sf.value("inner_parallelism", 0.0) * sf.value("outer_parallelism", 1.0);
            }
            features["total_parallelism"] = total_parallelism;

            features["bytes_per_vector"] = features["total_bytes_at_production"] / (features["total_vectors"] + 1e-8);
            features["memory_pressure"] = features["sched_working_set"] / (features["sched_bytes_at_production"] + 1e-8);
            features["bytes_per_parallelism"] = features["total_bytes_at_production"] / (features["total_parallelism"] + 1e-8);
            features["nodes_per_schedule"] = features["nodes_count"] / (features["scheduling_count"] + 1e-8);
        }

        features["op_diversity"] = std::count_if(op_counts.begin(), op_counts.end(), [](const auto& pair) { return pair.second > 0; });

        features["computation_efficiency"] = features["sched_points_computed_total"] / (features["execution_time_ms"] + 1e-8);
        features["bytes_processing_rate"] = features["total_bytes_at_production"] / (features["execution_time_ms"] + 1e-8);
        features["memory_utilization_ratio"] = features["sched_working_set"] / (features["sched_bytes_at_production"] + 1e-8);

        // Create fixed-length feature vector
        std::map<std::string, double> fixed_features;
        for (const auto& key : FIXED_FEATURES) {
            fixed_features[key] = features.count(key) ? features.at(key) : 0.0;
        }
        return fixed_features;
    } catch (const std::exception& e) {
        std::cerr << "Error in feature extraction: " << e.what() << std::endl;
        return {};
    }
}

// Apply RobustScaler transformation
torch::Tensor robust_scale(const torch::Tensor& data, const std::vector<double>& center, const std::vector<double>& scale) {
    if (data.size(1) != static_cast<int64_t>(center.size()) || center.size() != scale.size()) {
        throw std::runtime_error("Dimension mismatch in robust_scale: data_dim=" + std::to_string(data.size(1)) +
                                 ", center_dim=" + std::to_string(center.size()));
    }
    auto data_acc = data.accessor<float, 2>();
    std::vector<float> scaled_data(data.numel());
    for (int64_t i = 0; i < data.size(0); ++i) {
        for (int64_t j = 0; j < data.size(1); ++j) {
            scaled_data[i * data.size(1) + j] = (data_acc[i][j] - center[j]) / (scale[j] + 1e-8);
        }
    }
    return torch::tensor(scaled_data, torch::kFloat).reshape({data.size(0), data.size(1)});
}

// Preprocess data (mirrors Python's prepare_data_for_model)
std::tuple<torch::Tensor, torch::Tensor> preprocess_data(
    const std::map<std::string, double>& features,
    const json& metadata,
    const std::vector<double>& scaler_node_center,
    const std::vector<double>& scaler_node_scale,
    const std::vector<double>& scaler_scalar_center,
    const std::vector<double>& scaler_scalar_scale
) {
    try {
        // Sequence preprocessing
        if (!metadata.contains("max_sequence_length") || !metadata.contains("seq_input_size")) {
            throw std::runtime_error("Missing metadata keys: max_sequence_length or seq_input_size");
        }
        int64_t sequence_length = metadata["max_sequence_length"].get<int64_t>();
        int64_t seq_input_size = metadata["seq_input_size"].get<int64_t>();
        if (seq_input_size != static_cast<int64_t>(FIXED_FEATURES.size())) {
            throw std::runtime_error("Mismatch in seq_input_size: metadata=" + std::to_string(seq_input_size) +
                                     ", FIXED_FEATURES=" + std::to_string(FIXED_FEATURES.size()));
        }
        std::vector<float> seq_data(seq_input_size);
        for (size_t i = 0; i < FIXED_FEATURES.size(); ++i) {
            const auto& key = FIXED_FEATURES[i];
            seq_data[i] = features.count(key) ? static_cast<float>(features.at(key)) : 0.0f;
            if (!features.count(key)) {
                std::cerr << "Warning: Missing sequence feature: " << key << std::endl;
            }
        }

        // Create sequence by repeating features
        std::vector<float> seq_data_padded(sequence_length * seq_input_size);
        for (int64_t i = 0; i < sequence_length; ++i) {
            std::copy(seq_data.begin(), seq_data.end(), seq_data_padded.begin() + i * seq_input_size);
        }
        torch::Tensor seq_tensor = torch::tensor(scaled_data, torch::kFloat).reshape({1, sequence_length, seq_input_size});

        // Scale sequence features
        torch::Tensor seq_scaled = robust_scale(seq_tensor.reshape({-1, seq_input_size}), scaler_node_center, scaler_node_scale);
        seq_scaled = seq_scaled.reshape({1, sequence_length, seq_input_size});

        // Scalar preprocessing
        if (!metadata.contains("scalar_features") || !metadata.contains("dropped_features") || !metadata.contains("skewed_features")) {
            throw std::runtime_error("Missing metadata keys: scalar_features, dropped_features, or skewed_features");
        }
        auto scalar_features = metadata["scalar_features"].get<std::vector<std::string>>();
        auto dropped_features = metadata["dropped_features"].get<std::vector<std::string>>();
        auto skewed_features = metadata["skewed_features"].get<std::vector<std::string>>();

        // Create scalar feature vector
        std::vector<float> scalar_data;
        std::vector<std::string> final_scalar_features;
        for (const auto& feature : scalar_features) {
            if (std::find(dropped_features.begin(), dropped_features.end(), feature) == dropped_features.end()) {
                if (!features.count(feature)) {
                    std::cerr << "Warning: Missing scalar feature: " << feature << std::endl;
                    scalar_data.push_back(0.0f);
                    final_scalar_features.push_back(feature);
                    continue;
                }
                if (std::find(skewed_features.begin(), skewed_features.end(), feature) != skewed_features.end()) {
                    scalar_data.push_back(std::log1p(features.at(feature)));
                    final_scalar_features.push_back("log_" + feature);
                } else {
                    scalar_data.push_back(static_cast<float>(features.at(feature)));
                    final_scalar_features.push_back(feature);
                }
            }
        }

        // Verify scalar feature dimensions
        if (scalar_data.size() != scaler_scalar_center.size()) {
            throw std::runtime_error("Scalar feature dimension mismatch: data=" + std::to_string(scalar_data.size()) +
                                     ", scaler=" + std::to_string(scaler_scalar_center.size()));
        }

        // Remove constant columns (if any, based on metadata)
        torch::Tensor scalar_tensor = torch::tensor(scalar_data, torch::kFloat).reshape({1, -1});

        // Scale scalar features
        torch::Tensor scalar_scaled = robust_scale(scalar_tensor, scaler_scalar_center, scaler_scalar_scale);
        scalar_scaled = torch::nan_to_num(scalar_scaled, 0.0);

        return {seq_scaled, scalar_scaled};
    } catch (const std::exception& e) {
        std::cerr << "Error in preprocessing: " << e.what() << std::endl;
        return {torch::Tensor(), torch::Tensor()};
    }
}

// Perform inference
double perform_inference(
    torch::jit::script::Module& model,
    const torch::Tensor& seq_input,
    const torch::Tensor& scalar_input,
    const std::vector<double>& y_center,
    const std::vector<double>& y_scale,
    const torch::Device& device
) {
    if (seq_input.numel() == 0 || scalar_input.numel() == 0) {
        throw std::runtime_error("Invalid input tensors for inference");
    }
    try {
        model.eval();
        auto seq_input_device = seq_input.to(device);
        auto scalar_input_device = scalar_input.to(device);

        std::vector<torch::jit::IValue> inputs = {seq_input_device, scalar_input_device};
        auto output = model.forward(inputs).toTensor();

        // Inverse transform output
        auto output_acc = output.accessor<float, 2>();
        float scaled_pred = output_acc[0][0];
        float transformed_pred = (scaled_pred * y_scale[0]) + y_center[0];
        double actual_pred = std::expm1(transformed_pred);
        return std::max(actual_pred, 0.0);
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_converted_function_graph.json>" << std::endl;
        return 1;
    }

    try {
        // Determine device
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available. Using GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        } else {
            std::cout << "Using CPU." << std::endl;
        }

        // Load model
        std::cout << "Loading model from model.pt" << std::endl;
        torch::jit::script::Module model = torch::jit::load("model.pt", device);
        model.to(device);
        std::cout << "Model loaded" << std::endl;

        // Load metadata and scaler parameters
        std::cout << "Loading metadata from model_metadata.json" << std::endl;
        auto metadata = load_model_metadata("model_metadata.json");
        std::cout << "Metadata loaded: seq_input_size=" << metadata["seq_input_size"].get<int64_t>()
                  << ", scalar_input_size=" << metadata["scalar_input_size"].get<int64_t>() << std::endl;

        std::cout << "Loading scaler from scaler_node_params.json" << std::endl;
        auto [scaler_node_center, scaler_node_scale] = load_scaler_params("scaler_node_params.json");
        std::cout << "Scaler loaded: center size=" << scaler_node_center.size()
                  << ", scale size=" << scaler_node_scale.size() << std::endl;

        std::cout << "Loading scaler from scaler_scalar_params.json" << std::endl;
        auto [scaler_scalar_center, scaler_scalar_scale] = load_scaler_params("scaler_scalar_params.json");
        std::cout << "Scaler loaded: center size=" << scaler_scalar_center.size()
                  << ", scale size=" << scaler_scalar_scale.size() << std::endl;

        std::cout << "Loading scaler from scaler_y_params.json" << std::endl;
        auto [y_center, y_scale] = load_scaler_params("scaler_y_params.json");
        std::cout << "Scaler loaded: center size=" << y_center.size()
                  << ", scale size=" << y_scale.size() << std::endl;

        // Load and process input JSON
        std::string input_file = argv[1];
        std::cout << "Input file: " << input_file << std Valdation of input file existence
        if (!std::filesystem::exists(input_file)) {
            throw std::runtime_error("Input file does not exist: " + input_file);
        }
        std::cout << "Input JSON loaded" << std::endl;

        std::cout << "Extracting features from JSON" << std::endl;
        json json_data = load_json(input_file);
        auto features = extract_features(json_data);
        if (features.empty()) {
            throw std::runtime_error("Failed to extract valid features from " + input_file);
        }
        std::cout << "Features extracted: " << features.size() << " features" << std::endl;

        // Preprocess data
        std::cout << "Preprocessing data" << std::endl;
        auto [seq_tensor, scalar_tensor] = preprocess_data(
            features, metadata,
            scaler_node_center, scaler_node_scale,
            scaler_scalar_center, scaler_scalar_scale
        );

        if (seq_tensor.numel() == 0 || scalar_tensor.numel() == 0) {
            throw std::runtime_error("Preprocessing failed: empty tensors");
        }
        std::cout << "Data preprocessed: seq_tensor shape=[" << seq_tensor.sizes() << "], scalar_tensor shape=[" << scalar_tensor.sizes() << "]" << std::endl;

        // Perform inference
        std::cout << "Performing inference" << std::endl;
        double prediction = perform_inference(model, seq_tensor, scalar_tensor, y_center, y_scale, device);
        if (prediction < 0) {
            throw std::runtime_error("Inference failed: invalid prediction");
        }

        std::cout << "Predicted execution time: " << prediction << " ms" << std::endl;
        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}
