cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(LSTMInference)


# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to the libtorch CMake config
set(Torch_DIR "/home/kowrisaan/jathu/libtorch/share/cmake/Torch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find libtorch
find_package(Torch REQUIRED)

# Link libtorch
target_link_libraries(test PRIVATE "${TORCH_LIBRARIES}")

# Include torch headers
target_include_directories(test PRIVATE "${TORCH_INCLUDE_DIRS}")

# Ensure ABI compatibility
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)


# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Add executable
add_executable(lstm_inference test.cpp)

# Link libraries
target_link_libraries(lstm_inference "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)

# Ensure the binary can find LibTorch shared libraries
if (APPLE)
    set_target_properties(lstm_inference PROPERTIES
        BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
        INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib"
    )
endif()
