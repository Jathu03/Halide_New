cmake_minimum_required(VERSION 3.10)
project(HalideInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
set(Torch_DIR "/home/kowrisaan/jathu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find nlohmann_json
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Add executable
add_executable(test graph.cpp)

# Link libraries
target_link_libraries(test "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)

# Ensure CUDA is available
if(TORCH_CUDA_LIBRARIES)
    target_link_libraries(test ${TORCH_CUDA_LIBRARIES})
endif()
