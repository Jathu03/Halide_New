cmake_minimum_required(VERSION 3.10)
project(LSTMInference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to LibTorch
set(CMAKE_PREFIX_PATH "/home/kowrisaan/jathu/libtorch")  # Use absolute path for clarity

# Find LibTorch
find_package(Torch REQUIRED)
if(Torch_FOUND)
    message(STATUS "LibTorch found: ${TORCH_LIBRARIES}")
    message(STATUS "LibTorch include dirs: ${TORCH_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "LibTorch not found. Please check CMAKE_PREFIX_PATH.")
endif()

# Include nlohmann/json (assuming it's in 'Halide_New/json')
include_directories(${CMAKE_SOURCE_DIR}/json)

# Include LibTorch headers
include_directories(${TORCH_INCLUDE_DIRS})

# Add executable
add_executable(lstm_inference hello.cpp)

# Link LibTorch libraries
target_link_libraries(lstm_inference PRIVATE ${TORCH_LIBRARIES})

# Print compiler flags for debugging
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
