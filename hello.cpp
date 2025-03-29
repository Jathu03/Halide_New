#include <torch/script.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

std::vector<float> read_csv(const std::string &file_path) {
    std::vector<float> input_data;
    std::ifstream file(file_path);
    std::string line, value;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, value, ',')) {  // Assuming CSV is comma-separated
            input_data.push_back(std::stof(value));
        }
    }
    return input_data;
}

int main() {
    // Load the trained model
    torch::jit::script::Module model = torch::jit::load("halide_lstm_variant.pt");

    // Read input data from CSV file
    std::vector<float> input_data = read_csv("input_features.csv");

    // Convert to tensor (adjust {1, seq_len, input_size} as per your model)
    auto input_tensor = torch::from_blob(input_data.data(), {1, seq_len, input_size});

    // Run inference
    auto output = model.forward({input_tensor}).toTensor();
    std::cout << "Predicted execution time: " << output << std::endl;

    return 0;
}
