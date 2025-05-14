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
        center = params["center"].get<std::vector
