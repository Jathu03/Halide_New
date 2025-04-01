cmake_minimum_required(VERSION 3.10)
project(LSTMInference)

# Set the path to LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../../jathu/libtorch")

# Find LibTorch
find_package(Torch REQUIRED)

# Add nlohmann/json include directory (assuming it's in 'json' subfolder)
include_directories(${CMAKE_SOURCE_DIR}/json)

# Add executable (assuming the file is named hello.cpp)
add_executable(lstm_inference hello.cpp)

# Link libraries
target_link_libraries(lstm_inference "${TORCH_LIBRARIES}")

# Set C++ standard
set_property(TARGET lstm_inference PROPERTY CXX_STANDARD 17)

# Optional: Ensure CUDA is enabled if available
if(TORCH_CUDA_LIBRARIES)
    target_link_libraries(lstm_inference "${TORCH_CUDA_LIBRARIES}")
endif()
