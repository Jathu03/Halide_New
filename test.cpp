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

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono;

// Constants
constexpr size_t SEQUENCE_LENGTH = 3;
constexpr double EPSILON = 1e-8;

constexpr std::array<const char*, 66> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    // Add your remaining 62 feature names here
};

// Correction Factors
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
            extract_global_features(json_data, features);
            extract_op_histogram(json_data, features);
            extract_memory_patterns(json_data, features);
            extract_scheduling_features(json_data, features);
            compute_derived_features(features);
        } catch (const std::exception& e) {
            std::cerr << "Feature extraction error: " << e.what() << '\n';
        }

        return features;
    }

private:
    static void extract_global_features(const json& j, std::unordered_map<std::string, double>& f) {
        for (const auto& item : j.items()) {
            if (f.find(item.key()) != f.end() && item.value().is_number()) {
                f[item.key()] = item.value().get<double>();
            }
        }
    }

    static void extract_op_histogram(const json& j, std::unordered_map<std::string, double>& f) {
        if (j.contains("op_histogram")) {
            for (const auto& [op, count] : j["op_histogram"].items()) {
                if (count.is_number()) {
                    f["op_" + op] = count.get<double>();
                }
            }
        }
    }

    static void extract_memory_patterns(const json& j, std::unordered_map<std::string, double>& f) {
        if (j.contains("memory_accesses")) {
            for (const auto& [pattern, access] : j["memory_accesses"].items()) {
                if (access.is_number()) {
                    f["mem_" + pattern] = access.get<double>();
                }
            }
        }
    }

    static void extract_scheduling_features(const json& j, std::unordered_map<std::string, double>& f) {
        if (j.contains("scheduling")) {
            for (const auto& [feat, val] : j["scheduling"].items()) {
                if (val.is_number()) {
                    f["sched_" + feat] = val.get<double>();
                }
            }
        }
    }

    static void compute_derived_features(std::unordered_map<std::string, double>& f) {
        f["cache_hit_ratio"] = f["cache_hits"] / (f["cache_hits"] + f["cache_misses"] + EPSILON);
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
        double confidence = 0.0;
        confidence += features.at("cache_hit_ratio");
        confidence = std::clamp(confidence, 0.0, 1.0);
        return confidence;
    }

    static double blend_with_actual(double predicted, double actual, const HardwareCorrectionFactors& factors) {
        double weight = factors.confidence_threshold;
        return weight * actual + (1.0 - weight) * predicted;
    }
};

// PredictionModel
class PredictionModel {
public:
    PredictionModel(const std::string& model_path, torch::Device device)
        : device_(device) {
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
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

            if (torch::isnan(prediction).any().item<bool>() || 
                torch::isinf(prediction).any().item<bool>()) {
                throw std::runtime_error("Invalid model output.");
            }

            return prediction.item<double>();
        } catch (const c10::Error&) {
            if (device_.is_cuda()) {
                return fallback_to_cpu(seq_input, scalar_input, y_center, y_scale);
            }
            throw;
        }
    }

private:
    torch::jit::Module model_;
    torch::Device device_;

    double fallback_to_cpu(torch::Tensor seq_input, torch::Tensor scalar_input,
                           double y_center, double y_scale) {
        torch::Device cpu_device(torch::kCPU);
        model_.to(cpu_device);
        seq_input = seq_input.to(cpu_device);
        scalar_input = scalar_input.to(cpu_device);

        auto output = model_.forward({seq_input, scalar_input}).toTensor();
        auto transformed = output * y_scale + y_center;
        auto prediction = torch::expm1(transformed);

        if (torch::isnan(prediction).any().item<bool>() || 
            torch::isinf(prediction).any().item<bool>()) {
            throw std::runtime_error("CPU fallback failed.");
        }

        return prediction.item<double>();
    }
};

// HalidePredictor
class HalidePredictor {
public:
    HalidePredictor(const std::string& model_path,
                    const std::string& scaler_path,
                    const std::string& calibration_path = "")
        : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          model_(model_path, device_),
          factors_(device_.is_cuda() ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS) {

        load_scaler_params(scaler_path);
        if (!calibration_path.empty()) {
            calibration_data_ = load_calibration_data(calibration_path);
        }
    }

    void process_files(const std::vector<std::string>& files) {
        auto start = high_resolution_clock::now();
        size_t count = 0;

        for (const auto& file : files) {
            try {
                process_single_file(file);
                count++;
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << file << ": " << e.what() << '\n';
            }
        }

        auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
        std::cout << "Processed " << count << "/" << files.size()
                  << " files in " << duration.count() << "ms\n";
    }

    void save_calibration_data(const std::string& path) {
        std::ofstream ofs(path);
        for (const auto& [file, params] : calibration_data_) {
            ofs << file << " " << params.first << " " << params.second << "\n";
        }
    }

private:
    torch::Device device_;
    PredictionModel model_;
    HardwareCorrectionFactors factors_;
    std::unordered_map<std::string, double> scalar_centers_;
    std::unordered_map<std::string, double> scalar_scales_;
    std::unordered_map<std::string, std::pair<double, double>> calibration_data_;

    void load_scaler_params(const std::string& path) {
        std::ifstream ifs(path);
        json j;
        ifs >> j;

        for (const auto& [k, v] : j.items()) {
            scalar_centers_[k] = v["center"];
            scalar_scales_[k] = v["scale"];
        }
    }

    std::unordered_map<std::string, std::pair<double, double>> load_calibration_data(const std::string& path) {
        std::unordered_map<std::string, std::pair<double, double>> calibration;
        std::ifstream ifs(path);
        std::string file;
        double scale, bias;

        while (ifs >> file >> scale >> bias) {
            calibration[file] = {scale, bias};
        }
        return calibration;
    }

    void process_single_file(const std::string& file) {
        std::ifstream ifs(file);
        json j;
        ifs >> j;

        auto features = FeatureExtractor::extract(j);

        torch::Tensor seq_input = torch::zeros({1, SEQUENCE_LENGTH, static_cast<int64_t>(features.size())});
        torch::Tensor scalar_input = torch::zeros({1, static_cast<int64_t>(features.size())});

        int idx = 0;
        for (const auto& [k, v] : features) {
            double center = scalar_centers_.count(k) ? scalar_centers_.at(k) : 0.0;
            double scale = scalar_scales_.count(k) ? scalar_scales_.at(k) : 1.0;
            scalar_input[0][idx] = (v - center) / (scale + EPSILON);
            idx++;
        }

        double y_center = 0.0, y_scale = 1.0;
        double raw_pred = model_.predict(seq_input, scalar_input, y_center, y_scale);

        double corrected = PredictionCorrector::correct(
            raw_pred, features["execution_time_ms"], device_.is_cuda(), factors_, calibration_data_, file, features
        );

        std::cout << "Prediction for " << file << ": " << corrected << " ms\n";
    }
};

std::vector<std::string> load_test_files(const std::string& filename) {
    std::ifstream ifs(filename);
    std::vector<std::string> files;
    std::string line;
    while (std::getline(ifs, line)) {
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
            std::cerr << "No input files specified.\n";
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
