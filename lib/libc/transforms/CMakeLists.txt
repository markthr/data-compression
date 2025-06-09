pybind11_add_module(Transforms transforms.cpp)

target_include_directories(Transforms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Transforms PUBLIC test_compiler_flags)