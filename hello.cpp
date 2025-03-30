#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    // Load the traced model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("halide_lstm.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return -1;
    }

    // Example input (replace with real Halide features)
    std::vector<float> features = { /* Your extracted features */ };
    torch::Tensor input_tensor = torch::from_blob(
        features.data(), {1, 1, features.size()}, torch::kFloat32);

    // Run inference
    torch::Tensor output = model.forward({input_tensor}).toTensor();
    float predicted_time = output.item<float>();

    std::cout << "Predicted execution time: " << predicted_time << " ms\n";
    return 0;
}
