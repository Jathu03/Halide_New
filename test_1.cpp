#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>

using json = nlohmann::json;

// RobustScaler implementation
struct RobustScaler {
    std::vector<float> center;
    std::vector<float> scale;

    void load(const std::string& params_file) {
        std::cout << "Loading scaler from " << params_file << std::endl;
        std::ifstream file(params_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open " + params_file);
        }
        json params;
        file >> params;
        center = params["center"].get<std::vector<float>>();
        scale = params["scale"].get<std::vector<float>>();
        std::cout << "Scaler loaded: center size=" << center.size() << ", scale size=" << scale.size() << std::endl;
    }

    std::vector<float> transform(const std::vector<float>& input) {
        if (input.size() != center.size()) {
            throw std::runtime_error("Input size (" + std::to_string(input.size()) + ") does not match scaler size (" + std::to_string(center.size()) + ")");
        }
        std::vector<float> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = (input[i] - center[i]) / (scale[i] + 1e-8);
        }
