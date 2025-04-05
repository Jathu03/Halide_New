# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Project name and language
project(MyLibTorchProject LANGUAGES CXX)

# Set C++ standard to C++17 (required for std::filesystem and nlohmann_json)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Torch package (LibTorch)
# Set the path to your LibTorch installation
set(CMAKE_PREFIX_PATH "C:/libtorch")
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found. Please set CMAKE_PREFIX_PATH to the LibTorch installation directory.")
endif()

# Find nlohmann_json package
find_package(nlohmann_json 3.10.5 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found. Fetching and building nlohmann_json...")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Add the executable target
add_executable(MyLibTorchProject main.cpp)

# Include directories for the target
target_include_directories(MyLibTorchProject PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

# Link libraries to the target
target_link_libraries(MyLibTorchProject PRIVATE
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# On Windows, link against filesystem library if needed (for std::filesystem)
if(WIN32)
    target_link_libraries(MyLibTorchProject PRIVATE bcrypt)
endif()

# Set compiler flags for MSVC (if using Visual Studio/MSBuild)
if(MSVC)
    target_compile_options(MyLibTorchProject PRIVATE /W4 /WX)
    # Ensure LibTorch DLLs are found at runtime by copying them to the output directory
    foreach(lib ${TORCH_LIBRARIES})
        get_filename_component(lib_name ${lib} NAME_WE)
        get_filename_component(lib_dir ${lib} DIRECTORY)
        add_custom_command(TARGET MyLibTorchProject POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${lib_dir}/${lib_name}.dll"
                "$<TARGET_FILE_DIR:MyLibTorchProject>"
            COMMENT "Copying ${lib_name}.dll to build directory"
        )
    endforeach()
endif()

# Copy model and scaler files to the build directory if they exist
foreach(file "lstm_model.pt" "scaler_X.json" "scaler_y.json" "0_0.json")
    if(EXISTS "${CMAKE_SOURCE_DIR}/${file}")
        add_custom_command(TARGET MyLibTorchProject POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/${file}"
                "$<TARGET_FILE_DIR:MyLibTorchProject>"
            COMMENT "Copying ${file} to build directory"
        )
    else()
        message(WARNING "File ${file} not found in ${CMAKE_SOURCE_DIR}. Skipping copy.")
    endif()
endforeach()

# Set properties for the target (e.g., to ensure LibTorch uses CPU)
set_property(TARGET MyLibTorchProject PROPERTY CXX_STANDARD 17)
set_property(TARGET torch PROPERTY INTERFACE_COMPILE_DEFINITIONS "TORCH_API_INCLUDE_EXTENSION_H")
