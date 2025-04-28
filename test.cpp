#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <array>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Constants
constexpr std::array<std::string_view, 74> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", /* ... other features ... */
    "memory_pointwise_3"
};

// Hardware correction factors
struct HardwareCorrectionFactors {
    double base_correction = 0.28;
    double gpu_correction = 0.9;
    double scaling_factor = 0.95;
    double min_time_ms = 100.0;
    double high_threshold_ms = 500.0;
    double high_scaling = 0.92;
};

// Category correction
struct CategoryCorrection {
    double scale_factor = 1.0;
    double bias = 0.0;
    double confidence = 0.7;
    int sample_count = 0;
};

// Configuration
struct Config {
    std::string model_path = "model.pt";
    std::string scaler_params_path = "scaler_params.json";
    std::string calibration_path = "calibration_data.txt";
    std::string category_calibration_path = "category_calibration.txt";
    std::string test_files_path = "test_files.txt";
    int sequence_length = 3;
    bool verbose_logging = false;
};

// Logger
class Logger {
public:
    static void init(bool verbose) {
        auto console = spdlog::stdout_color_mt("console");
        spdlog::set_default_logger(console);
        spdlog::set_level(verbose ? spdlog::level::debug : spdlog::level::info);
    }
};

// Feature Extractor
class FeatureExtractor {
public:
    static std::unordered_map<std::string, double> extract(const json& data) {
        std::unordered_map<std::string, double> features;
        // Extract global features
        auto global_node = std::find_if(data["children"].begin(), data["children"].end(),
            [](const auto& child) { return child["name"] == "Global Features"; });
        if (global_node != data["children"].end()) {
            features["cache_hits"] = global_node->value("cache_hits", 0.0);
            features["cache_misses"] = global_node->value("cache_misses", 0.0);
            features["execution_time_ms"] = global_node->value("execution_time_ms", 0.0);
        }
        // Add other feature extractions (op_histogram, memory_patterns, etc.)
        // ... (implement similar to original code, but use unordered_map)
        return features;
    }
};

// Predictor class
class Predictor {
public:
    Predictor(const Config& config) : config_(config) {
        Logger::init(config_.verbose_logging);
        load_model();
        load_scaler_params();
        load_calibration_data();
    }

    void process_files(const std::vector<std::string>& files) {
        std::mutex result_mutex;
        std::vector<std::future<void>> futures;

        for (const auto& file : files) {
            futures.push_back(std::async(std::launch::async, [this, &file, &result_mutex]() {
                process_single_file(file, result_mutex);
            }));
        }

        for (auto& future : futures) {
            future.wait();
        }

        print_results();
        save_calibration_data();
    }

private:
    void load_model() {
        try {
            model_ = torch::jit::load(config_.model_path);
            model_.to(device_);
            model_.eval();
            spdlog::info("Model loaded successfully from {}", config_.model_path);
        } catch (const c10::Error& e) {
            spdlog::error("Failed to load model: {}", e.what());
            throw;
        }
    }

    void load_scaler_params() {
        std::ifstream scaler_file(config_.scaler_params_path);
        if (!scaler_file.is_open()) {
            spdlog::error("Failed to open scaler params: {}", config_.scaler_params_path);
            throw std::runtime_error("Scaler params not found");
        }
        scaler_file >> scaler_params_;
        // Load scaler parameters
        // ... (implement similar to original code)
    }

    void load_calibration_data() {
        // Load file-specific and category-specific calibration
        // ... (implement similar to original code, but use unordered_map)
    }

    void process_single_file(const std::string& file, std::mutex& result_mutex) {
        json json_data;
        try {
            std::ifstream json_file(file);
            if (!json_file.is_open()) {
                spdlog::warn("Failed to open file: {}", file);
                return;
            }
            json_file >> json_data;
        } catch (const json::exception& e) {
            spdlog::warn("Error parsing JSON file {}: {}", file, e.what());
            return;
        }

        auto features = FeatureExtractor::extract(json_data);
        std::string category = get_file_category(file, features);
        double actual_time = features["execution_time_ms"];

        // Prepare inputs
        auto [seq_input, scalar_input] = prepare_inputs(features);

        // Get prediction
        double raw_prediction = get_raw_prediction(seq_input, scalar_input);
        if (raw_prediction < 0) {
            spdlog::warn("Failed to get prediction for {}", file);
            return;
        }

        // Correct prediction
        double corrected_prediction = correct_prediction(raw_prediction, actual_time, file, category, features);

        // Update calibration
        if (actual_time > 0) {
            update_calibration(file, category, raw_prediction, actual_time);
        }

        // Store result
        std::lock_guard<std::mutex> lock(result_mutex);
        results_by_category_[category].emplace_back(file, actual_time, raw_prediction, corrected_prediction);
    }

    std::pair<torch::Tensor, torch::Tensor> prepare_inputs(const std::unordered_map<std::string, double>& features) {
        // Prepare sequence and scalar inputs
        // ... (implement similar to original code)
        return {torch::Tensor(), torch::Tensor()};
    }

    double get_raw_prediction(const torch::Tensor& seq_input, const torch::Tensor& scalar_input) {
        // Implement model inference with fallback
        // ... (implement similar to original code)
        return 0.0;
    }

    double correct_prediction(double raw_prediction, double actual_time, const std::string& file,
                             const std::string& category, const std::unordered_map<std::string, double>& features) {
        // Implement prediction correction
        // ... (implement similar to original code)
        return raw_prediction;
    }

    void update_calibration(const std::string& file, const std::string& category,
                           double raw_prediction, double actual_time) {
        // Implement calibration update
        // ... (implement similar to original code)
    }

    void print_results() {
        // Print results by category and overall metrics
        // ... (implement similar to original code)
    }

    void save_calibration_data() {
        // Save calibration data
        // ... (implement similar to original code)
    }

    std::string get_file_category(const std::string& file, const std::unordered_map<std::string, double>& features) {
        // Implement category determination
        // ... (implement similar to original code)
        return "unknown";
    }

private:
    Config config_;
    torch::jit::script::Module model_;
    torch::Device device_{torch::cuda::is_available() ? torch::kCUDA : torch::kCPU};
    json scaler_params_;
    std::unordered_map<std::string, std::pair<double, double>> file_calibration_;
    std::unordered_map<std::string, CategoryCorrection> category_calibration_;
    std::unordered_map<std::string, std::vector<std::tuple<std::string, double, double, double>>> results_by_category_;
};

int main(int argc, char* argv[]) {
    try {
        Config config;
        if (argc > 1) {
            // Load files from command line
            std::vector<std::string> test_files(argv + 1, argv + argc);
            Predictor predictor(config);
            predictor.process_files(test_files);
        } else {
            // Load from test_files.txt
            std::vector<std::string> test_files;
            std::ifstream file(config.test_files_path);
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) test_files.push_back(line);
            }
            if (test_files.empty()) {
                spdlog::error("No test files found in {}", config.test_files_path);
                return 1;
            }
            Predictor predictor(config);
            predictor.process_files(test_files);
        }
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }
    spdlog::info("Prediction process complete.");
    return 0;
}
