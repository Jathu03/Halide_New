cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_test)

# Point to the libtorch CMake config
set(Torch_DIR "/home/kowrisaan/jathu/libtorch/share/cmake/Torch")

# Find libtorch
find_package(Torch REQUIRED)

# Your executable
add_executable(main test.cpp)

# Link libtorch
target_link_libraries(main "${TORCH_LIBRARIES}")

# Include torch headers
target_include_directories(main PUBLIC "${TORCH_INCLUDE_DIRS}")

# Set the C++ standard
set_property(TARGET main PROPERTY CXX_STANDARD 14)
