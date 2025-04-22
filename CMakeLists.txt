cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_test)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} /home/kowrisaan/jathu/libtorch")

add_executable(main test.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
