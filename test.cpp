#include <torch/torch.h>
#include <iostream>

int main() {
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Running on CPU." << std::endl;
    }

    // Create a tensor on CPU
    torch::Tensor tensor = torch::rand({3, 3});
    std::cout << "CPU Tensor:\n" << tensor << std::endl;

    // Move tensor to GPU if available
    if (torch::cuda::is_available()) {
        tensor = tensor.to(torch::kCUDA);
        std::cout << "Tensor moved to GPU:\n" << tensor << std::endl;
    }

    return 0;
}
