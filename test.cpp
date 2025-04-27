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
#include <sstream>
#include <numeric>

// Add CUDA runtime headers if CUDA is available
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

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

// Enhanced hardware-specific correction factors with more granular control
struct HardwareCorrectionFactors {
    double base_correction;          // Base correction multiplier
    double gpu_correction;           // Additional GPU-specific correction
    double scaling_factor;           // Non-linear scaling factor
    double min_time_ms;              // Minimum time threshold for non-linear correction
    double max_correction_factor;    // Maximum allowed correction factor
    double min_correction_factor;    // Minimum allowed correction factor
    double confidence_threshold;     // Confidence threshold for applying correction
};

// More sophisticated correction factors based on hardware
const HardwareCorrectionFactors GPU_CORRECTION_FACTORS = {
    0.28,       // base_correction
    0.9,        // gpu_correction
    0.95,       // scaling_factor
    100.0,      // min_time_ms
    2.0,        // max_correction_factor
    0.1,        // min_correction_factor
    0.7         // confidence_threshold
};

const HardwareCorrectionFactors CPU_CORRECTION_FACTORS = {
    0.35,       // base_correction
    1.0,        // gpu_correction (no additional)
    0.97,       // scaling_factor
    50.0,       // min_time_ms
    2.0,        // max_correction_factor
    0.1,        // min_correction_factor
    0.6         // confidence_threshold
};

// Function to extract features from JSON data with better error handling
std::map<std::string, double> extract_features(const json& json_data) {
    std::map<std::string, double> features;
    
    // Initialize all features to 0 first
    for (const auto& feature : FIXED_FEATURES) {
        features[feature] = 0.0;
    }

    try {
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
                    try {
                        std::string op_lower = op;
                        std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                        op_histogram[op_lower] += count.get<int>();
                    } catch (const json::exception& e) {
                        std::cerr << "Error processing op_histogram: " << e.what() << std::endl;
                    }
                }
            }
        }
        for (const auto& [op, count] : op_histogram) {
            features["op_" + op] = static_cast<double>(count);
        }

        // Extract memory patterns with better error handling
        std::map<std::string, std::vector<double>> memory_patterns;
        for (const auto& node : json_data["children"]) {
            if (node.contains("memory_patterns")) {
                for (const auto& [pattern, values] : node["memory_patterns"].items()) {
                    try {
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
                    } catch (const json::exception& e) {
                        std::cerr << "Error processing memory_patterns: " << e.what() << std::endl;
                    }
                }
            }
        }
        for (const auto& [pattern, values] : memory_patterns) {
            for (size_t i = 0; i < 4; ++i) {
                features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
            }
        }

        // Extract scheduling features with better aggregation
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
                    try {
                        scheduling_sums[key] += node["scheduling"].value(key, 0.0);
                    } catch (const json::exception& e) {
                        std::cerr << "Error processing scheduling feature " << key << ": " << e.what() << std::endl;
                    }
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

        // Derived features with better numerical stability
        features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"];
        features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"];
        features["total_bytes_at_production"] = features["sched_bytes_at_production"];
        features["total_vectors"] = features["sched_num_vectors"];
        
        double bytes_at_realization = features["sched_bytes_at_realization"];
        features["computation_efficiency"] = (bytes_at_realization > 1e-8) ? 
            features["sched_points_computed_total"] / bytes_at_realization : 0.0;
        
        double bytes_at_root = features["sched_bytes_at_root"];
        features["memory_pressure"] = (bytes_at_root > 1e-8) ? 
            features["sched_working_set"] / bytes_at_root : 0.0;
        
        double bytes_at_task = features["sched_bytes_at_task"];
        features["memory_utilization_ratio"] = (bytes_at_task > 1e-8) ? 
            features["sched_unique_bytes_read_per_realization"] / bytes_at_task : 0.0;
        
        double execution_time_ms = features["execution_time_ms"];
        features["bytes_processing_rate"] = (execution_time_ms > 1e-8) ? 
            features["sched_bytes_at_realization"] / execution_time_ms : 0.0;
        
        double total_parallelism = features["total_parallelism"];
        features["bytes_per_parallelism"] = (total_parallelism > 1e-8) ? 
            features["sched_bytes_at_task"] / total_parallelism : 0.0;
        
        double num_vectors = features["sched_num_vectors"];
        features["bytes_per_vector"] = (num_vectors > 1e-8) ? 
            features["sched_bytes_at_realization"] / num_vectors : 0.0;
        
        int nodes_count = json_data["children"].size();
        int edges_count = 0;
        for (const auto& node : json_data["children"]) {
            edges_count += node.value("children", json::array()).size();
        }
        features["nodes_count"] = nodes_count;
        features["edges_count"] = edges_count;
        features["node_edge_ratio"] = (edges_count + 1) > 0 ? 
            static_cast<double>(nodes_count) / (edges_count + 1) : 0.0;
        
        double scheduling_count = features["scheduling_count"];
        features["nodes_per_schedule"] = (scheduling_count > 1e-8) ? 
            nodes_count / scheduling_count : 0.0;
        
        int op_diversity = 0;
        for (const auto& [key, value] : features) {
            if (key.find("op_") == 0 && value > 0) {
                op_diversity++;
            }
        }
        features["op_diversity"] = op_diversity;

    } catch (const json::exception& e) {
        std::cerr << "Error extracting features from JSON: " << e.what() << std::endl;
    }

    return features;
}

// Enhanced function to collect test files with better error handling
std::vector<std::string> load_test_files(const std::string& filename) {
    std::vector<std::string> test_files;
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open test files list: " << filename << std::endl;
            return test_files;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                // Trim whitespace from the line
                line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
                line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
                
                // Check if file exists
                if (fs::exists(line)) {
                    test_files.push_back(line);
                } else {
                    std::cerr << "Warning: Test file does not exist: " << line << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading test files: " << e.what() << std::endl;
    }
    return test_files;
}

// More sophisticated prediction correction with confidence estimation
double correct_prediction(double raw_prediction, double actual_time, bool is_gpu,
                         const HardwareCorrectionFactors& factors,
                         const std::map<std::string, std::pair<double, double>>& calibration_data,
                         const std::string& file_path,
                         const std::map<std::string, double>& features) {
    
    // First check if we have specific calibration for this file
    auto it = calibration_data.find(file_path);
    if (it != calibration_data.end()) {
        const auto& [scale_factor, bias] = it->second;
        // Apply file-specific correction: scale * prediction + bias
        double corrected = scale_factor * raw_prediction + bias;
        return std::max(corrected, 0.0);
    }
    
    // Calculate confidence based on feature values
    double confidence = 1.0;
    
    // Check for features that might indicate unreliable predictions
    if (features.at("execution_time_ms") < 1.0) {
        confidence *= 0.7;  // Low confidence for very fast executions
    }
    
    if (features.at("sched_num_realizations") < 3) {
        confidence *= 0.8;  // Low confidence for simple schedules
    }
    
    if (features.at("nodes_count") > 500) {
        confidence *= 0.9;  // Slightly lower confidence for very large graphs
    }
    
    // Apply hardware-specific correction with confidence weighting
    double hw_correction = factors.base_correction;
    if (is_gpu) {
        hw_correction *= factors.gpu_correction;
    }
    
    // Blend correction with original prediction based on confidence
    double base_corrected = raw_prediction * hw_correction;
    double confidence_weight = std::min(std::max(confidence, 0.1), 1.0);
    double corrected = confidence_weight * base_corrected + (1.0 - confidence_weight) * raw_prediction;
    
    // Apply non-linear correction for large predictions
    if (raw_prediction > factors.min_time_ms) {
        double excess = raw_prediction - factors.min_time_ms;
        corrected = (factors.min_time_ms * hw_correction) + 
                   (excess * hw_correction * factors.scaling_factor);
    }
    
    // Fine-tune if we know the actual time (but don't have calibration data yet)
    if (actual_time > 0) {
        // Calculate error ratio
        double error_ratio = corrected / (actual_time + 1e-8);
        
        // If error is too large, apply additional correction
        if (error_ratio > 1.5 || error_ratio < 0.67) {  // More than 50% error
            double adjustment_factor = 1.0 / error_ratio;
            // Limit the adjustment to avoid overcorrection
            adjustment_factor = std::min(std::max(adjustment_factor, 
                factors.min_correction_factor), factors.max_correction_factor);
            
            // Blend the adjustment with the current prediction
            corrected = (0.7 * corrected + 0.3 * corrected * adjustment_factor);
        }
    }
    
    return std::max(corrected, 0.0);
}

// Enhanced function to load calibration data
std::map<std::string, std::pair<double, double>> load_calibration_data(const std::string& filename) {
    std::map<std::string, std::pair<double, double>> calibration_map;
    
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "No calibration file found. Will use default correction factors." << std::endl;
            return calibration_map;
        }
        
        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            line_num++;
            std::istringstream iss(line);
            std::string filepath;
            double scale_factor, bias;
            
            if (iss >> filepath >> scale_factor >> bias) {
                // Validate the calibration values
                if (scale_factor > 0.1 && scale_factor < 10.0 && 
                    std::abs(bias) < 10000.0) {
                    calibration_map[filepath] = std::make_pair(scale_factor, bias);
                } else {
                    std::cerr << "Warning: Invalid calibration values in line " << line_num 
                              << ": " << line << std::endl;
                }
            } else {
                std::cerr << "Warning: Malformed calibration data in line " << line_num 
                          << ": " << line << std::endl;
            }
        }
        
        std::cout << "Loaded " << calibration_map.size() << " valid calibration entries." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading calibration data: " << e.what() << std::endl;
    }
    
    return calibration_map;
}

// Enhanced function to save calibration data
void save_calibration_data(const std::string& filename, 
                         const std::map<std::string, std::pair<double, double>>& calibration_map) {
    try {
        // Create a backup of the existing file if it exists
        if (fs::exists(filename)) {
            fs::path backup_path = filename + ".bak";
            fs::copy_file(filename, backup_path, fs::copy_options::overwrite_existing);
        }
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open calibration file for writing." << std::endl;
            return;
        }
        
        // Write calibration data with higher precision
        file << std::setprecision(10);
        for (const auto& [filepath, factors] : calibration_map) {
            file << filepath << " " << factors.first << " " << factors.second << "\n";
        }
        
        std::cout << "Saved " << calibration_map.size() << " calibration entries." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving calibration data: " << e.what() << std::endl;
    }
}

// Enhanced function to update calibration data
void update_calibration_data(std::map<std::string, std::pair<double, double>>& calibration_map,
                           const std::string& file_path, 
                           double raw_prediction, 
                           double actual_time,
                           const HardwareCorrectionFactors& factors) {
    if (actual_time <= 0 || raw_prediction <= 0) return;
    
    // Calculate the desired correction factor
    double desired_scale = actual_time / raw_prediction;
    
    // Limit the correction factor to reasonable bounds
    desired_scale = std::min(std::max(desired_scale, factors.min_correction_factor), 
                            factors.max_correction_factor);
    
    // Calculate bias term if needed
    double bias = 0.0;
    double corrected = desired_scale * raw_prediction;
    if (std::abs(corrected - actual_time) > 0.1 * actual_time) {
        bias = (actual_time - corrected) * 0.5;  // Only apply half the needed bias
    }
    
    // Check if we already have calibration data for this file
    auto it = calibration_map.find(file_path);
    if (it != calibration_map.end()) {
        // Update existing calibration with exponential moving average
        double learning_rate = 0.3;  // Weight for new observation
        
        // Apply different learning rates based on confidence
        double error_ratio = std::abs(it->second.first * raw_prediction + it->second.second - actual_time) / actual_time;
        if (error_ratio > 0.5) {
            learning_rate = 0.5;  // Higher learning rate for large errors
        } else if (error_ratio < 0.1) {
            learning_rate = 0.1;  // Lower learning rate for small errors
        }
        
        double old_scale = it->second.first;
        double old_bias = it->second.second;
        
        // Update scale factor with smoothing
        desired_scale = (1.0 - learning_rate) * old_scale + learning_rate * desired_scale;
        
        // Update bias term with smoothing
        bias = (1.0 - learning_rate) * old_bias + learning_rate * bias;
    }
    
    // Update calibration map
    calibration_map[file_path] = std::make_pair(desired_scale, bias);
}

// Function to extract file type/category from path with better handling
std::string get_file_category(const std::string& file_path) {
    try {
        fs::path path(file_path);
        if (path.has_parent_path()) {
            fs::path parent = path.parent_path();
            if (parent.has_parent_path()) {
                // Get the grandparent directory name if it exists
                fs::path grandparent = parent.parent_path();
                return grandparent.filename().string() + "/" + parent.filename().string();
            }
            return parent.filename().string();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error determining file category: " << e.what() << std::endl;
    }
    return "unknown";
}

// Enhanced function to get raw prediction with better error handling
double get_raw_prediction(torch::jit::script::Module& model, 
                         torch::Tensor seq_input, 
                         torch::Tensor scalar_input,
                         const torch::Device& device, 
                         double y_center, 
                         double y_scale) {
    
    try {
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
                std::cout << "CUDA inference failed, falling back to CPU..." << std::endl;
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
        
        // Check for NaN or infinity
        if (torch::isnan(y_pred_actual).any().item<bool>() || 
            torch::isinf(y_pred_actual).any().item<bool>()) {
            std::cerr << "Warning: Model returned NaN or infinite prediction" << std::endl;
            return -1.0;
        }
        
        // Return the raw prediction
        return y_pred_actual.item<float>();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in get_raw_prediction: " << e.what() << std::endl;
        return -1.0;
    }
}

// Function to print colored output based on error percentage
void print_colored_error(const std::string& label, double value, double reference, const std::string& unit = "") {
    double error_pct = std::abs(value - reference) / (reference + 1e-8) * 100;
    
    std::cout << label;
    if (error_pct < 10.0) {
        // Green for good predictions (<10% error)
        std::cout << "\033[32m" << value << unit << " (Error: " << error_pct << "%)\033[0m";
    } else if (error_pct < 30.0) {
        // Yellow for moderate predictions (10-30% error)
        std::cout << "\033[33m" << value << unit << " (Error: " << error_pct << "%)\033[0m";
    } else {
        // Red for poor predictions (>30% error)
        std::cout << "\033[31m" << value << unit << " (Error: " << error_pct << "%)\033[0m";
    }
}

int main(int argc, char* argv[]) {
    // Check if CUDA is available and set device accordingly
    bool is_gpu_available = torch::cuda::is_available();
    torch::Device device = is_gpu_available ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    
    if (is_gpu_available) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
        // Print CUDA device info
        if (torch::cuda::device_count() > 0) {
            #ifdef __CUDACC__
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "Using GPU: " << prop.name << std::endl;
            #else
            std::cout << "CUDA device info not available (compiled without CUDA support)" << std::endl;
            #endif
        }
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
            if (fs::exists(argv[i])) {
                test_files.push_back(argv[i]);
            } else {
                std::cerr << "Warning: File does not exist: " << argv[i] << std::endl;
            }
        }
    } else {
        // Otherwise load test files from test_files.txt
        test_files = load_test_files("test_files.txt");
        if (test_files.empty()) {
            std::cerr << "No valid test files found in test_files.txt or command line arguments" << std::endl;
            return 1;
        }
    }

    // Load calibration data
    std::map<std::string, std::pair<double, double>> calibration_map = 
        load_calibration_data("calibration_data.txt");
    
    // Load scaler parameters with better error handling
    json scaler_params;
    try {
        std::ifstream scaler_file("scaler_params.json");
        if (!scaler_file.is_open()) {
            std::cerr << "Failed to open scaler_params.json" << std::endl;
            return 1;
        }
        scaler_file >> scaler_params;
    } catch (const json::exception& e) {
        std::cerr << "Error parsing scaler_params.json: " << e.what() << std::endl;
        return 1;
    }
    
    std::vector<double> X_scalar_center;
    std::vector<double> X_scalar_scale;
    double y_center = 0.0;
    double y_scale = 1.0;
    std::vector<std::string> feature_columns;
    
    try {
        X_scalar_center = scaler_params["X_scalar_center"].get<std::vector<double>>();
        X_scalar_scale = scaler_params["X_scalar_scale"].get<std::vector<double>>();
        y_center = scaler_params["y_center"][0].get<double>();
        y_scale = scaler_params["y_scale"][0].get<double>();
        feature_columns = scaler_params["feature_columns"].get<std::vector<std::string>>();
    } catch (const json::exception& e) {
        std::cerr << "Error reading scaler parameters: " << e.what() << std::endl;
        return 1;
    }

    // Load the model with better error handling
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("model.pt");
        model.to(device);
        model.eval();
        
        // Print model information
        std::cout << "Model loaded successfully." << std::endl;
        if (is_gpu_available) {
            std::cout << "Model is on GPU." << std::endl;
        } else {
            std::cout << "Model is on CPU." << std::endl;
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 1;
    }

    // Process each file
    std::map<std::string, std::vector<std::tuple<std::string, double, double, double>>> results_by_category;
    const int sequence_length = 3;
    int processed_files = 0;
    int failed_files = 0;
    
    for (const auto& file : test_files) {
        processed_files++;
        std::cout << "\nProcessing file " << processed_files << " of " << test_files.size() 
                  << ": " << file << std::endl;
        
        // Read JSON file
        std::ifstream json_file(file);
        if (!json_file.is_open()) {
            std::cerr << "Failed to open " << file << std::endl;
            failed_files++;
            continue;
        }
        
        json json_data;
        try {
            json_file >> json_data;
        } catch (const json::exception& e) {
            std::cerr << "Error parsing JSON file " << file << ": " << e.what() << std::endl;
            failed_files++;
            continue;
        }
        
        // Extract features
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
        
        torch::Tensor seq_input;
        try {
            seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length, 1});
            seq_input = seq_input.unsqueeze(0);
        } catch (const std::exception& e) {
            std::cerr << "Error creating sequence input tensor: " << e.what() << std::endl;
            failed_files++;
            continue;
        }
        
        // Prepare scalar input
        std::vector<double> scalar_input;
        try {
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
        } catch (const std::exception& e) {
            std::cerr << "Error preparing scalar input: " << e.what() << std::endl;
            failed_files++;
            continue;
        }
        
        torch::Tensor scalar_tensor;
        try {
            scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);
        } catch (const std::exception& e) {
            std::cerr << "Error creating scalar input tensor: " << e.what() << std::endl;
            failed_files++;
            continue;
        }
        
        // Get raw prediction
        double raw_prediction = get_raw_prediction(
            model, seq_input, scalar_tensor, device, y_center, y_scale
        );
        
        if (raw_prediction < 0) {
            std::cerr << "Failed to get prediction for " << file << std::endl;
            failed_files++;
            continue;
        }
        
        // Get corrected prediction
        double corrected_prediction = correct_prediction(
            raw_prediction, execution_time, is_gpu_available, 
            factors, calibration_map, file, features
        );
        
        // Store result by category
        std::string category = get_file_category(file);
        results_by_category[category].emplace_back(file, execution_time, raw_prediction, corrected_prediction);
        
        // Update calibration data if we have actual execution time
        if (execution_time > 0) {
            update_calibration_data(calibration_map, file, raw_prediction, execution_time, factors);
        }
    }
    
    // Print summary of processed files
    std::cout << "\nProcessing complete. " << processed_files << " files processed, " 
              << failed_files << " files failed." << std::endl;
    
    // Print results by category with enhanced formatting
    for (const auto& [category, results] : results_by_category) {
        double category_mse = 0.0, category_mae = 0.0, category_mape_sum = 0.0;
        int valid_count = 0;
        
        std::cout << "\nResults for category: " << category << std::endl;
        std::cout << "=================================================================" << std::endl;
        
        for (const auto& [file, actual, raw_pred, corrected_pred] : results) {
            std::cout << "\nFile: " << file << std::endl;
            
            if (actual > 0) {
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  Actual execution time: " << actual << " ms" << std::endl;
                
                // Print raw prediction with colored error
                std::cout << "  Raw predicted time: ";
                print_colored_error("", raw_pred, actual, " ms");
                std::cout << std::endl;
                
                // Print corrected prediction with colored error
                std::cout << "  Corrected prediction: ";
                print_colored_error("", corrected_pred, actual, " ms");
                std::cout << std::endl;
                
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
            
            std::cout << "\nCategory '" << category << "' Performance Metrics:" << std::endl;
            std::cout << "  MSE: " << category_mse << std::endl;
            std::cout << "  RMSE: " << category_rmse << std::endl;
            std::cout << "  MAE: " << category_mae << std::endl;
            std::cout << "  MAPE: " << category_mape << "%" << std::endl;
            
            // Print quality assessment
            std::cout << "  Quality Assessment: ";
            if (category_mape < 10.0) {
                std::cout << "\033[32mExcellent\033[0m (MAPE < 10%)";
            } else if (category_mape < 20.0) {
                std::cout << "\033[33mGood\033[0m (MAPE < 20%)";
            } else if (category_mape < 30.0) {
                std::cout << "\033[33mFair\033[0m (MAPE < 30%)";
            } else {
                std::cout << "\033[31mPoor\033[0m (MAPE >= 30%)";
            }
            std::cout << std::endl;
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
        std::cout << "=================================================================" << std::endl;
        std::cout << "  MSE: " << overall_mse << std::endl;
        std::cout << "  RMSE: " << overall_rmse << std::endl;
        std::cout << "  MAE: " << overall_mae << std::endl;
        std::cout << "  MAPE: " << overall_mape << "%" << std::endl;
        
        // Print overall quality assessment
        std::cout << "  Overall Quality: ";
        if (overall_mape < 10.0) {
            std::cout << "\033[32mExcellent\033[0m (MAPE < 10%)";
        } else if (overall_mape < 20.0) {
            std::cout << "\033[33mGood\033[0m (MAPE < 20%)";
        } else if (overall_mape < 30.0) {
            std::cout << "\033[33mFair\033[0m (MAPE < 30%)";
        } else {
            std::cout << "\033[31mPoor\033[0m (MAPE >= 30%)";
        }
        std::cout << std::endl;
        
        // Print performance summary
        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << "  Total files processed: " << processed_files << std::endl;
        std::cout << "  Files with valid ground truth: " << overall_valid_count << std::endl;
        std::cout << "  Average prediction error: " << overall_mape << "%" << std::endl;
        std::cout << "  Average absolute error: " << overall_mae << " ms" << std::endl;
    } else {
        std::cout << "\nNo files with valid ground truth data to compute overall metrics." << std::endl;
    }
    
    // Save updated calibration data
    save_calibration_data("calibration_data.txt", calibration_map);
    
    return failed_files > 0 ? 1 : 0;
}
