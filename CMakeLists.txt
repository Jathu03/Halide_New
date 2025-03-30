cmake_minimum_required(VERSION 3.10)
project(LSTMInference)

# Set the path to LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../../jathu/libtorch")

# Find LibTorch
find_package(Torch REQUIRED)

# Add nlohmann/json (assuming it's in 'Halide_New/json')
include_directories(${CMAKE_SOURCE_DIR}/json)

# Add executable
add_executable(lstm_inference hello.cpp)
target_link_libraries(lstm_inference "${TORCH_LIBRARIES}")
set_property(TARGET lstm_inference PROPERTY CXX_STANDARD 17)
