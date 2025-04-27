#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <memory>
#include <chrono>

// Conditional CUDA includes
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono;

// Constants
constexpr size_t SEQUENCE_LENGTH = 3;
constexpr double EPSILON = 1e-8;  // Small value to prevent division by zero

// Feature names - using constexpr array for better performance
constexpr std::array<const char*, 66> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    // ... [rest of your feature names]
};

// Enhanced hardware correction factors with constexpr
struct HardwareCorrectionFactors {
    const double base_correction;
    const double gpu_correction;
    const double scaling_factor;
    const double min_time_ms;
    const double max_correction_factor;
    const double min_correction_factor;
    const double confidence_threshold;
};

// Using constexpr for compile-time constants
constexpr HardwareCorrectionFactors GPU_CORRECTION_FACTORS = {
    0.28, 0.9, 0.95, 100.0, 2.0, 0.1, 0.7
};

constexpr HardwareCorrectionFactors CPU_CORRECTION_FACTORS = {
    0.35, 1.0, 0.97, 50.0, 2.0, 0.1, 0.6
};

// Feature extraction with optimizations
class FeatureExtractor {
public:
    static std::unordered_map<std::string, double> extract(const json& json_data) {
        std::unordered_map<std::string, double> features;
        features.reserve(FIXED_FEATURES.size());

        // Initialize all features to 0
        for (const auto& feature : FIXED_FEATURES) {
            features[feature] = 0.0;
        }

        try {
            extract_global_features(json_data, features);
            extract_op_histogram(json_data, features);
            extract_memory_patterns(json_data, features);
            extract_scheduling_features(json_data, features);
            compute_derived_features(features);
        } catch (const json::exception& e) {
            std::cerr << "JSON error in feature extraction: " << e.what() << '\n';
        } catch (const std::exception& e) {
            std::cerr << "Error in feature extraction: " << e.what() << '\n';
        }

        return features;
    }

private:
    // [Private helper methods for each extraction step]
};

// Prediction corrector with improved algorithms
class PredictionCorrector {
public:
    static double correct(double raw_prediction, double actual_time, bool is_gpu,
                        const HardwareCorrectionFactors& factors,
                        const std::unordered_map<std::string, std::pair<double, double>>& calibration_data,
                        const std::string& file_path,
                        const std::unordered_map<std::string, double>& features) {
        
        // Check for calibration data first
        if (auto it = calibration_data.find(file_path); it != calibration_data.end()) {
            const auto& [scale, bias] = it->second;
            return std::max(scale * raw_prediction + bias, 0.0);
        }

        // Calculate confidence based on features
        double confidence = calculate_confidence(features);
        
        // Apply hardware-specific correction
        double correction = factors.base_correction * (is_gpu ? factors.gpu_correction : 1.0);
        double corrected = raw_prediction * correction;

        // Apply confidence weighting
        corrected = confidence * corrected + (1.0 - confidence) * raw_prediction;

        // Non-linear correction for large predictions
        if (raw_prediction > factors.min_time_ms) {
            double excess = raw_prediction - factors.min_time_ms;
            corrected = (factors.min_time_ms * correction) + 
                       (excess * correction * factors.scaling_factor);
        }

        // Fine-tuning with actual time if available
        if (actual_time > 0) {
            corrected = blend_with_actual(corrected, actual_time, factors);
        }

        return std::max(corrected, 0.0);
    }

private:
    // [Private helper methods]
};

// Model wrapper for better resource management
class PredictionModel {
public:
    PredictionModel(const std::string& model_path, torch::Device device) 
        : device_(device) {
        try {
            model_ = torch::jit::load(model_path);
            model_.to(device_);
            model_.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Failed to load model: " + std::string(e.what()));
        }
    }

    double predict(torch::Tensor seq_input, torch::Tensor scalar_input,
                  double y_center, double y_scale) {
        torch::NoGradGuard no_grad;
        
        try {
            // Move inputs to device
            seq_input = seq_input.to(device_);
            scalar_input = scalar_input.to(device_);

            // Run inference
            auto output = model_.forward({seq_input, scalar_input}).toTensor();
            
            // Transform and validate output
            auto transformed = output * y_scale + y_center;
            auto prediction = torch::expm1(transformed);

            if (torch::isnan(prediction).any().item<bool>() || 
                torch::isinf(prediction).any().item<bool>()) {
                throw std::runtime_error("Model returned invalid prediction");
            }

            return prediction.item<double>();
        } catch (const c10::Error& e) {
            if (device_.is_cuda()) {
                return fallback_to_cpu(seq_input, scalar_input, y_center, y_scale);
            }
            throw std::runtime_error("Prediction failed: " + std::string(e.what()));
        }
    }

private:
    torch::jit::Module model_;
    torch::Device device_;

    double fallback_to_cpu(torch::Tensor seq_input, torch::Tensor scalar_input,
                          double y_center, double y_scale) {
        // [CPU fallback implementation]
    }
};

// Main application class
class HalidePredictor {
public:
    HalidePredictor(const std::string& model_path, 
                   const std::string& scaler_params_path,
                   const std::string& calibration_path = "")
        : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          model_(model_path, device_),
          factors_(device_.is_cuda() ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS) {
        
        load_scaler_params(scaler_params_path);
        if (!calibration_path.empty()) {
            calibration_data_ = load_calibration_data(calibration_path);
        }
    }

    void process_files(const std::vector<std::string>& files) {
        auto start = high_resolution_clock::now();
        size_t processed = 0;

        for (const auto& file : files) {
            try {
                process_single_file(file);
                processed++;
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << file << ": " << e.what() << '\n';
            }
        }

        auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
        std::cout << "Processed " << processed << "/" << files.size() 
                  << " files in " << duration.count() << "ms\n";
    }

    void save_calibration_data(const std::string& path) {
        // [Implementation]
    }

private:
    // [Private member functions and variables]
};

int main(int argc, char* argv[]) {
    try {
        // Initialize with paths
        HalidePredictor predictor("model.pt", "scaler_params.json", "calibration_data.txt");

        // Get input files
        std::vector<std::string> input_files;
        if (argc > 1) {
            input_files.assign(argv + 1, argv + argc);
        } else {
            input_files = load_test_files("test_files.txt");
        }

        if (input_files.empty()) {
            std::cerr << "No input files specified\n";
            return 1;
        }

        // Process files
        predictor.process_files(input_files);
        predictor.save_calibration_data("calibration_data.txt");

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
