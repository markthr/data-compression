cmake_minimum_required(VERSION 3.12)
project(DataCompression)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(pybind11 REQUIRED)
set(pybind11_FINDPYTHON ON)

add_library(test_compiler_flags INTERFACE)
target_compile_options(test_compiler_flags INTERFACE
    "-Wall"
    "-Wextra"
)
target_compile_features(test_compiler_flags INTERFACE cxx_std_20)
add_subdirectory(transforms)
add_subdirectory(test)


