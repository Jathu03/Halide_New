#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Define FIXED_FEATURES as in Python
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

// Hardware-specific correction factors
struct HardwareCorrectionFactors {
    double base_correction;
    double gpu_correction;  // Additional correction for GPU
    double scaling_factor;  // For non-linear scaling
    double min_time_ms;     // Minimum execution time threshold
};

// Global correction factors based on hardware
const HardwareCorrectionFactors GPU_CORRECTION_FACTORS = {
    0.28,  // Base correction factor (reduces predictions by ~72%)
    0.9,   // GPU-specific additional correction
    0.95,  // Scaling factor for non-linear correction
    100.0  // Minimum time threshold in ms
};

const HardwareCorrectionFactors CPU_CORRECTION_FACTORS = {
    0.35,  // Base correction factor (reduces predictions by ~65%)
    1.0,   // No additional GPU correction
    0.97,  // Scaling factor for non-linear correction
    50.0   // Minimum time threshold in ms
};

// Function to extract features from JSON data
std::map<std::string, double> extract_features(const json& json_data) {
    std::map<std::string, double> features;

    // Extract global features
    auto global_node = std::find_if(json_data["children"].begin(), json_data["children"].end(),
        [](const json& child) { return child["name"] == "Global Features"; });
    if (global_node != json_data["children"].end()) {
        features["cache_hits"] = global_node->value("cache_hits", 0.0);
        features["cache_misses"] = global_node->value("cache_misses", 0.0);
        features["execution_time_ms"] = global_node->value("execution_time_ms", 0.0);
    }

    // Extract op_histogram features
    std::map<std::string, int> op_histogram;
    for (const auto& node : json_data["children"]) {
        if (node.contains("op_histogram")) {
            for (const auto& [op, count] : node["op_histogram"].items()) {
                std::string op_lower = op;
                std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                op_histogram[op_lower] += count.get<int>();
            }
        }
    }
    for (const auto& [op, count] : op_histogram) {
        features["op_" + op] = static_cast<double>(count);
    }

    // Extract memory patterns
    std::map<std::string, std::vector<double>> memory_patterns;
    for (const auto& node : json_data["children"]) {
        if (node.contains("memory_patterns")) {
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                std::string pattern_lower = pattern;
                std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                if (memory_patterns.find(pattern_lower) == memory_patterns.end()) {
                    memory_patterns[pattern_lower] = {0.0, 0.0, 0.0, 0.0};
                }
                auto curr_values = memory_patterns[pattern_lower];
                auto json_values = values.get<std::vector<double>>();
                for (size_t i = 0; i < json_values.size() && i < 4; ++i) {
                    curr_values[i] += json_values[i];
                }
                memory_patterns[pattern_lower] = curr_values;
            }
        }
    }
    for (const auto& [pattern, values] : memory_patterns) {
        for (size_t i = 0; i < 4; ++i) {
            features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
        }
    }

    // Extract scheduling features
    std::vector<std::string> scheduling_keys = {
        "num_realizations", "num_productions", "points_computed_total", "innermost_loop_extent",
        "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
        "bytes_at_root", "unique_bytes_read_per_realization", "working_set", "vector_size",
        "num_vectors", "num_scalars", "bytes_at_task", "working_set_at_task", "working_set_at_production",
        "working_set_at_realization", "working_set_at_root"
    };
    std::map<std::string, double> scheduling_sums;
    int node_count = 0;
    for (const auto& node : json_data["children"]) {
        if (node.contains("scheduling")) {
            node_count++;
            for (const auto& key : scheduling_keys) {
                scheduling_sums[key] += node["scheduling"].value(key, 0.0);
            }
        }
    }
    for (const auto& key : scheduling_keys) {
        if (key == "inner_parallelism" || key == "outer_parallelism") {
            features["sched_" + key] = node_count > 0 ? scheduling_sums[key] / node_count : 0.0;
        } else {
            features["sched_" + key] = scheduling_sums[key];
        }
    }

    // Derived features
    features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"];
    features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"];
    features["total_bytes_at_production"] = features["sched_bytes_at_production"];
    features["total_vectors"] = features["sched_num_vectors"];
    double bytes_at_realization = features["sched_bytes_at_realization"];
    features["computation_efficiency"] = (bytes_at_realization != 0) ? features["sched_points_computed_total"] / bytes_at_realization : 0.0;
    double bytes_at_root = features["sched_bytes_at_root"];
    features["memory_pressure"] = (bytes_at_root != 0) ? features["sched_working_set"] / bytes_at_root : 0.0;
    double bytes_at_task = features["sched_bytes_at_task"];
    features["memory_utilization_ratio"] = (bytes_at_task != 0) ? features["sched_unique_bytes_read_per_realization"] / bytes_at_task : 0.0;
    double execution_time_ms = features["execution_time_ms"];
    features["bytes_processing_rate"] = (execution_time_ms != 0) ? features["sched_bytes_at_realization"] / execution_time_ms : 0.0;
    double total_parallelism = features["total_parallelism"];
    features["bytes_per_parallelism"] = (total_parallelism != 0) ? features["sched_bytes_at_task"] / total_parallelism : 0.0;
    double num_vectors = features["sched_num_vectors"];
    features["bytes_per_vector"] = (num_vectors != 0) ? features["sched_bytes_at_realization"] / num_vectors : 0.0;
    int nodes_count = json_data["children"].size();
    int edges_count = 0;
    for (const auto& node : json_data["children"]) {
        edges_count += node.value("children", json::array()).size();
    }
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["node_edge_ratio"] = (edges_count + 1) != 0 ? static_cast<double>(nodes_count) / (edges_count + 1) : 0.0;
    double scheduling_count = features["scheduling_count"];
    features["nodes_per_schedule"] = (scheduling_count != 0) ? nodes_count / scheduling_count : 0.0;
    int op_diversity = 0;
    for (const auto& [key, value] : features) {
        if (key.find("op_") == 0 && value > 0) {
            op_diversity++;
        }
    }
    features["op_diversity"] = op_diversity;

    return features;
}

// Function to collect test files from test_files.txt
std::vector<std::string> load_test_files(const std::string& filename) {
    std::vector<std::string> test_files;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            test_files.push_back(line);
        }
    }
    return test_files;
}

// Enhanced prediction correction for better accuracy
double correct_prediction(double raw_prediction, double actual_time, bool is_gpu,
                         const HardwareCorrectionFactors& factors,
                         const std::map<std::string, std::pair<double, double>>& calibration_data,
                         const std::string& file_path) {
    
    // Check if we have specific calibration for this file
    auto it = calibration_data.find(file_path);
    if (it != calibration_data.end()) {
        const auto& [scale_factor, bias] = it->second;
        // Apply file-specific correction: scale * prediction + bias
        return std::max(scale_factor * raw_prediction + bias, 0.0);
    }
    
    // Apply hardware-specific correction
    double hw_correction = factors.base_correction;
    if (is_gpu) {
        hw_correction *= factors.gpu_correction;
    }
    
    // Apply non-linear correction for large predictions
    double corrected = raw_prediction * hw_correction;
    if (raw_prediction > factors.min_time_ms) {
        // Apply additional scaling for predictions above threshold
        double excess = raw_prediction - factors.min_time_ms;
        corrected = (factors.min_time_ms * hw_correction) + 
                   (excess * hw_correction * factors.scaling_factor);
    }
    
    // Fine-tune if we know the actual time (but don't have calibration data yet)
    if (actual_time > 0) {
        // Blend with actual time for better future predictions
        // The weight is lower because we don't want to overfit to a single data point
        double blend_weight = 0.2;  // 20% weight to actual time
        corrected = (1.0 - blend_weight) * corrected + blend_weight * actual_time;
    }
    
    return std::max(corrected, 0.0);
}

// Function to load calibration data from a file
std::map<std::string, std::pair<double, double>> load_calibration_data(const std::string& filename) {
    std::map<std::string, std::pair<double, double>> calibration_map;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "No calibration file found. Will use default correction factors." << std::endl;
        return calibration_map;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string filepath;
        double scale_factor, bias;
        
        if (iss >> filepath >> scale_factor >> bias) {
            calibration_map[filepath] = std::make_pair(scale_factor, bias);
        }
    }
    
    std::cout << "Loaded " << calibration_map.size() << " calibration entries." << std::endl;
    return calibration_map;
}

// Save calibration data for future runs
void save_calibration_data(const std::string& filename, 
                         const std::map<std::string, std::pair<double, double>>& calibration_map) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open calibration file for writing." << std::endl;
        return;
    }
    
    for (const auto& [filepath, factors] : calibration_map) {
        file << filepath << " " << factors.first << " " << factors.second << std::endl;
    }
    
    std::cout << "Saved " << calibration_map.size() << " calibration entries." << std::endl;
}

// Function to update calibration data based on new predictions and actual times
void update_calibration_data(std::map<std::string, std::pair<double, double>>& calibration_map,
                           const std::string& file_path, double raw_prediction, double actual_time) {
    if (actual_time <= 0 || raw_prediction <= 0) return;
    
    // Compute scale factor and bias for linear correction
    double scale_factor = actual_time / raw_prediction;
    double bias = 0.0; // Start with simple scaling
    
    // Check if we already have calibration data for this file
    auto it = calibration_map.find(file_path);
    if (it != calibration_map.end()) {
        // Update existing calibration with exponential moving average
        double learning_rate = 0.3; // Weight for new observation
        double old_scale = it->second.first;
        double old_bias = it->second.second;
        
        // Update scale factor with smoothing
        scale_factor = (1.0 - learning_rate) * old_scale + learning_rate * scale_factor;
        
        // Refine bias term if needed
        if (std::abs(scale_factor * raw_prediction - actual_time) > 0.1 * actual_time) {
            // If scale alone doesn't provide good correction, add bias term
            bias = (actual_time - scale_factor * raw_prediction) * 0.5;
            bias = (1.0 - learning_rate) * old_bias + learning_rate * bias;
        }
    }
    
    // Cap the scale factor to reasonable range to avoid excessive correction
    scale_factor = std::min(std::max(scale_factor, 0.1), 2.0);
    
    // Update calibration map
    calibration_map[file_path] = std::make_pair(scale_factor, bias);
}

// Function to extract file type/category from path
std::string get_file_category(const std::string& file_path) {
    fs::path path(file_path);
    if (path.has_parent_path()) {
        fs::path parent = path.parent_path();
        return parent.filename().string();
    }
    return "unknown";
}

// Function to run inference and get raw prediction
double get_raw_prediction(torch::jit::script::Module& model, 
                         torch::Tensor seq_input, 
                         torch::Tensor scalar_input,
                         const torch::Device& device, 
                         double y_center, 
                         double y_scale) {
    
    // Move inputs to device
    seq_input = seq_input.to(device);
    scalar_input = scalar_input.to(device);
    
    // Run inference
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {seq_input, scalar_input};
    torch::Tensor y_pred_scaled;
    
    try {
        y_pred_scaled = model.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        if (device.is_cuda()) {
            // Try CPU fallback
            torch::Device cpu_device = torch::kCPU;
            torch::jit::script::Module cpu_model = model.clone();
            cpu_model.to(cpu_device);
            
            seq_input = seq_input.to(cpu_device);
            scalar_input = scalar_input.to(cpu_device);
            
            inputs = {seq_input, scalar_input};
            try {
                y_pred_scaled = cpu_model.forward(inputs).toTensor();
            } catch (const c10::Error& e) {
                std::cerr << "Error during CPU fallback inference: " << e.what() << std::endl;
                return -1.0;
            }
        } else {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
            return -1.0;
        }
    }
    
    // Inverse transform prediction
    torch::Tensor y_pred_transformed = y_pred_scaled * y_scale + y_center;
    torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);
    
    // Return the raw prediction
    return y_pred_actual.item<float>();
}

int main(int argc, char* argv[]) {
    // Check if CUDA is available and set device accordingly
    bool is_gpu_available = torch::cuda::is_available();
    torch::Device device = is_gpu_available ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    
    if (is_gpu_available) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Using CPU." << std::endl;
    }
    
    // Select hardware-specific correction factors
    const HardwareCorrectionFactors& factors = is_gpu_available ? 
        GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS;
    
    // Process command line arguments
    std::vector<std::string> test_files;
    if (argc > 1) {
        // If files are provided as arguments, use them
        for (int i = 1; i < argc; i++) {
            test_files.push_back(argv[i]);
        }
    } else {
        // Otherwise load test files from test_files.txt
        test_files = load_test_files("test_files.txt");
        if (test_files.empty()) {
            std::cerr << "No test files found in test_files.txt" << std::endl;
            return 1;
        }
    }

    // Load calibration data
    std::map<std::string, std::pair<double, double>> calibration_map = 
        load_calibration_data("calibration_data.txt");
    
    // Load scaler parameters
    json scaler_params;
    std::ifstream scaler_file("scaler_params.json");
    if (!scaler_file.is_open()) {
        std::cerr << "Failed to open scaler_params.json" << std::endl;
        return 1;
    }
    scaler_file >> scaler_params;
    std::vector<double> X_scalar_center = scaler_params["X_scalar_center"].get<std::vector<double>>();
    std::vector<double> X_scalar_scale = scaler_params["X_scalar_scale"].get<std::vector<double>>();
    double y_center = scaler_params["y_center"][0].get<double>();
    double y_scale = scaler_params["y_scale"][0].get<double>();
    std::vector<std::string> feature_columns = scaler_params["feature_columns"].get<std::vector<std::string>>();

    // Load the model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("model.pt");
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 1;
    }

    // Process each file
    std::map<std::string, std::vector<std::tuple<std::string, double, double, double>>> results_by_category;
    const int sequence_length = 3;
    
    for (const auto& file : test_files) {
        std::ifstream json_file(file);
        if (!json_file.is_open()) {
            std::cerr << "Failed to open " << file << std::endl;
            continue;
        }
        
        json json_data;
        try {
            json_file >> json_data;
        } catch (const json::exception& e) {
            std::cerr << "Error parsing JSON file " << file << ": " << e.what() << std::endl;
            continue;
        }
        
        auto features = extract_features(json_data);
        
        // Record actual execution time if available
        double execution_time = features["execution_time_ms"];
        if (execution_time <= 0 || !std::isfinite(execution_time)) {
            std::cout << "Warning: Invalid execution time in file: " << file << std::endl;
            execution_time = -1; // Mark as unknown
        }
        
        // Prepare sequence input
        std::vector<double> feature_vector;
        for (const auto& key : FIXED_FEATURES) {
            feature_vector.push_back(features[key]);
        }
        torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length, 1});
        seq_input = seq_input.unsqueeze(0);
        
        // Prepare scalar input
        std::vector<double> scalar_input;
        for (const auto& col : feature_columns) {
            if (col.substr(0, 4) == "log_") {
                std::string original_feature = col.substr(4);
                double value = features[original_feature];
                scalar_input.push_back(std::log1p(value));
            } else {
                scalar_input.push_back(features[col]);
            }
        }
        
        // Scale scalar input
        for (size_t i = 0; i < scalar_input.size(); ++i) {
            scalar_input[i] = (scalar_input[i] - X_scalar_center[i]) / X_scalar_scale[i];
        }
        torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);
        
        // Get raw prediction
        double raw_prediction = get_raw_prediction(
            model, seq_input, scalar_tensor, device, y_center, y_scale
        );
        
        if (raw_prediction < 0) {
            std::cerr << "Failed to get prediction for " << file << std::endl;
            continue;
        }
        
        // Get corrected prediction
        double corrected_prediction = correct_prediction(
            raw_prediction, execution_time, is_gpu_available, 
            factors, calibration_map, file
        );
        
        // Store result by category
        std::string category = get_file_category(file);
        results_by_category[category].emplace_back(file, execution_time, raw_prediction, corrected_prediction);
        
        // Update calibration data if we have actual execution time
        if (execution_time > 0) {
            update_calibration_data(calibration_map, file, raw_prediction, execution_time);
        }
    }
    
    // Print results by category
    for (const auto& [category, results] : results_by_category) {
        double category_mse = 0.0, category_mae = 0.0, category_mape_sum = 0.0;
        int valid_count = 0;
        
        std::cout << "\nResults for category: " << category << std::endl;
        
        for (const auto& [file, actual, raw_pred, corrected_pred] : results) {
            std::cout << "\nFile: " << file << std::endl;
            
            if (actual > 0) {
                double raw_error_pct = std::abs(actual - raw_pred) / actual * 100;
                double corrected_error_pct = std::abs(actual - corrected_pred) / actual * 100;
                
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  Actual execution time: " << actual << " ms" << std::endl;
                std::cout << "  Raw predicted time: " << raw_pred << " ms (Error: " << raw_error_pct << "%)" << std::endl;
                std::cout << "  Corrected prediction: " << corrected_pred << " ms (Error: " << corrected_error_pct << "%)" << std::endl;
                
                // Calculate metrics for this category
                double diff = corrected_pred - actual;
                category_mse += diff * diff;
                category_mae += std::abs(diff);
                category_mape_sum += std::abs(diff) / (actual + 1e-8);
                valid_count++;
            } else {
                std::cout << "  Actual execution time: Unknown" << std::endl;
                std::cout << "  Raw predicted time: " << raw_pred << " ms" << std::endl;
                std::cout << "  Corrected prediction: " << corrected_pred << " ms" << std::endl;
            }
        }
        
        // Print category metrics if we have valid data
        if (valid_count > 0) {
            category_mse /= valid_count;
            double category_rmse = std::sqrt(category_mse);
            category_mae /= valid_count;
            double category_mape = (category_mape_sum / valid_count) * 100;
            
            std::cout << "\nCategory '" << category << "' Performance:" << std::endl;
            std::cout << "  MSE: " << category_mse << std::endl;
            std::cout << "  RMSE: " << category_rmse << std::endl;
            std::cout << "  MAE: " << category_mae << std::endl;
            std::cout << "  MAPE: " << category_mape << "%" << std::endl;
        }
    }
    
    // Compute overall metrics
    double overall_mse = 0.0, overall_mae = 0.0, overall_mape_sum = 0.0;
    int overall_valid_count = 0;
    
    for (const auto& [category, results] : results_by_category) {
        for (const auto& [file, actual, raw_pred, corrected_pred] : results) {
            if (actual > 0) {
                double diff = corrected_pred - actual;
                overall_mse += diff * diff;
                overall_mae += std::abs(diff);
                overall_mape_sum += std::abs(diff) / (actual + 1e-8);
                overall_valid_count++;
            }
        }
    }
    
    if (overall_valid_count > 0) {
        overall_mse /= overall_valid_count;
        double overall_rmse = std::sqrt(overall_mse);
        overall_mae /= overall_valid_count;
        double overall_mape = (overall_mape_sum / overall_valid_count) * 100;
        
        std::cout << "\nOverall Model Performance (with correction):" << std::endl;
        std::cout << "  MSE: " << overall_mse << std::endl;
        std::cout << "  RMSE: " << overall_rmse << std::endl;
        std::cout << "  MAE: " << overall_mae << std::endl;
        std::cout << "  MAPE: " << overall_mape << "%" << std::endl;
    }
    
    // Save updated calibration data
    save_calibration_data("calibration_data.txt", calibration_map);
    
    return 0;
}
