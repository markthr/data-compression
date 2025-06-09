pybind11_add_module(transforms transforms.cpp)

target_include_directories(transforms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(transforms PUBLIC test_compiler_flags)