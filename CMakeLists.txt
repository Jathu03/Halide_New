cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(libtorch_test)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to the libtorch CMake config
set(Torch_DIR "/home/kowrisaan/jathu/libtorch/share/cmake/Torch")

# Find libtorch
find_package(Torch REQUIRED)

# Your executable
add_executable(test test.cpp)

# Link libtorch
target_link_libraries(test PRIVATE "${TORCH_LIBRARIES}")

# Include torch headers
target_include_directories(test PRIVATE "${TORCH_INCLUDE_DIRS}")

# Ensure ABI compatibility
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
