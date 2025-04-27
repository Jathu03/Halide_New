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
constexpr double EPSILON = 1e-8;

// Feature names - you should define all 66 features here
constexpr std::array<const char*, 66> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    // Add the rest of your features here
};

// Hardware Correction Factors
struct HardwareCorrectionFactors {
    const double base_correction;
    const double gpu_correction;
    const double scaling_factor;
    const double min_time_ms;
    const double max_correction_factor;
    const double min_correction_factor;
    const double confidence_threshold;
};

constexpr HardwareCorrectionFactors GPU_CORRECTION_FACTORS = {
    0.28, 0.9, 0.95, 100.0, 2.0, 0.1, 0.7
};

constexpr HardwareCorrectionFactors CPU_CORRECTION_FACTORS = {
    0.35, 1.0, 0.97, 50.0, 2.0, 0.1, 0.6
};

// FeatureExtractor
class FeatureExtractor {
public:
    static std::unordered_map<std::string, double> extract(const json& json_data) {
        std::unordered_map<std::string, double> features;
        features.reserve(FIXED_FEATURES.size());

        for (const auto& feature : FIXED_FEATURES) {
            features[feature] = 0.0;
        }

        try {
            if (json_data.is_array()) {
                for (const auto& element : json_data) {
                    extract_single(element, features);
                }
            } else {
                extract_single(json_data, features);
            }
            compute_derived_features(features);
        } catch (const json::exception& e) {
            std::cerr << "JSON error in feature extraction: " << e.what() << '\n';
        } catch (const std::exception& e) {
            std::cerr << "Error in feature extraction: " << e.what() << '\n';
        }

        return features;
    }

private:
    static void extract_single(const json& data, std::unordered_map<std::string, double>& features) {
        if (data.contains("execution_time_ms")) {
            features["execution_time_ms"] = data["execution_time_ms"].get<double>();
        }
        if (data.contains("cache_hits")) {
            features["cache_hits"] = data["cache_hits"].get<double>();
        }
        if (data.contains("cache_misses")) {
            features["cache_misses"] = data["cache_misses"].get<double>();
        }
        if (data.contains("sched_num_realizations")) {
            features["sched_num_realizations"] = data["sched_num_realizations"].get<double>();
        }
        // Add more feature extraction as needed
    }

    static void compute_derived_features(std::unordered_map<std::string, double>& features) {
        double total = features["cache_hits"] + features["cache_misses"];
        features["cache_hit_rate"] = total > EPSILON ? features["cache_hits"] / total : 0.0;
    }
};

// PredictionCorrector
class PredictionCorrector {
public:
    static double correct(double raw_prediction, double actual_time, bool is_gpu,
                        const HardwareCorrectionFactors& factors,
                        const std::unordered_map<std::string, std::pair<double, double>>& calibration_data,
                        const std::string& file_path,
                        const std::unordered_map<std::string, double>& features) {
        
        if (auto it = calibration_data.find(file_path); it != calibration_data.end()) {
            const auto& [scale, bias] = it->second;
            return std::max(scale * raw_prediction + bias, 0.0);
        }

        double confidence = calculate_confidence(features);
        double correction = factors.base_correction * (is_gpu ? factors.gpu_correction : 1.0);
        double corrected = raw_prediction * correction;

        corrected = confidence * corrected + (1.0 - confidence) * raw_prediction;

        if (raw_prediction > factors.min_time_ms) {
            double excess = raw_prediction - factors.min_time_ms;
            corrected = (factors.min_time_ms * correction) + (excess * correction * factors.scaling_factor);
        }

        if (actual_time > 0) {
            corrected = blend_with_actual(corrected, actual_time, factors);
        }

        return std::max(corrected, 0.0);
    }

private:
    static double calculate_confidence(const std::unordered_map<std::string, double>& features) {
        auto it = features.find("cache_hit_rate");
        if (it != features.end()) {
            return std::min(std::max(it->second, 0.0), 1.0);
        }
        return 0.5;
    }

    static double blend_with_actual(double corrected, double actual_time, const HardwareCorrectionFactors& factors) {
        double diff = std::abs(corrected - actual_time) / (actual_time + EPSILON);
        if (diff < factors.confidence_threshold) {
            return (corrected + actual_time) / 2.0;
        }
        return corrected;
    }
};

// PredictionModel
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
            seq_input = seq_input.to(device_);
            scalar_input = scalar_input.to(device_);

            auto output = model_.forward({seq_input, scalar_input}).toTensor();
            auto transformed = output * y_scale + y_center;
            auto prediction = torch::expm1(transformed);

            if (torch::isnan(prediction).any().item<bool>() || torch::isinf(prediction).any().item<bool>()) {
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
        model_.to(torch::kCPU);
        auto output = model_.forward({seq_input.to(torch::kCPU), scalar_input.to(torch::kCPU)}).toTensor();
        auto transformed = output * y_scale + y_center;
        auto prediction = torch::expm1(transformed);

        if (torch::isnan(prediction).any().item<bool>() || torch::isinf(prediction).any().item<bool>()) {
            throw std::runtime_error("Model returned invalid prediction in CPU fallback");
        }

        return prediction.item<double>();
    }
};

// HalidePredictor
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
        // Implement saving calibration data here if needed
    }

private:
    torch::Device device_;
    PredictionModel model_;
    HardwareCorrectionFactors factors_;
    std::unordered_map<std::string, std::pair<double, double>> calibration_data_;
    double y_center_ = 0.0;
    double y_scale_ = 1.0;

    void load_scaler_params(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open scaler params file");
        }
        json scaler_json;
        in >> scaler_json;
        y_center_ = scaler_json["center"].get<double>();
        y_scale_ = scaler_json["scale"].get<double>();
    }

    std::unordered_map<std::string, std::pair<double, double>> load_calibration_data(const std::string& path) {
        return {};  // Dummy for now
    }

    void process_single_file(const std::string& file) {
        std::ifstream in(file);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open input file: " + file);
        }
        json data;
        in >> data;

        auto features = FeatureExtractor::extract(data);

        torch::Tensor seq_input = torch::zeros({1, SEQUENCE_LENGTH, FIXED_FEATURES.size()});
        torch::Tensor scalar_input = torch::zeros({1, FIXED_FEATURES.size()});

        int idx = 0;
        for (const auto& feature : FIXED_FEATURES) {
            scalar_input[0][idx] = features.at(feature);
            idx++;
        }

        double raw_prediction = model_.predict(seq_input, scalar_input, y_center_, y_scale_);
        double corrected = PredictionCorrector::correct(raw_prediction, 0.0, device_.is_cuda(), factors_, calibration_data_, file, features);

        std::cout << "Predicted time for " << file << ": " << corrected << " ms\n";
    }
};

// Load list of test files
std::vector<std::string> load_test_files(const std::string& path) {
    std::vector<std::string> files;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        files.push_back(line);
    }
    return files;
}

int main(int argc, char* argv[]) {
    try {
        HalidePredictor predictor("model.pt", "scaler_params.json", "calibration_data.txt");

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

        predictor.process_files(input_files);
        predictor.save_calibration_data("calibration_data.txt");

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
