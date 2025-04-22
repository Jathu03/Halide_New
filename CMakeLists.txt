cmake_minimum_required(VERSION 3.10)
project(LSTMInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
set(Torch_DIR "/home/kowrisaan/jathu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
# Link libraries
target_link_libraries(lstm_inference "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)

# Ensure the binary can find LibTorch shared libraries
if (APPLE)
    set_target_properties(lstm_inference PROPERTIES
        BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
        INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib"
    )
endif()
