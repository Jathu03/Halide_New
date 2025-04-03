#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "CUDA available: " << torch::cuda::is_available() << "\n";
    return 0;
}
