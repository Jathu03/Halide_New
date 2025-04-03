#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch version: " << TORCH_VERSION << "\n";
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << "\n";
    return 0;
}
