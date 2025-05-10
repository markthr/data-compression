cmake_minimum_required(VERSION 3.12)
project(DataCompression)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(test_compiler_flags INTERFACE)
target_compile_options(test_compiler_flags INTERFACE
    "-Wall"
    "-Wextra"
)
target_compile_features(test_compiler_flags INTERFACE cxx_std_20)
add_subdirectory(transforms)
add_subdirectory(test)

# add_executable(placeholder transforms_test.cpp)
# target_link_libraries(placeholder PUBLIC Transforms)
# target_include_directories(placeholder PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/transforms")


