# Use this file to set the location of pybind11's CMake config file or use a different means to locate the library
set(pybind11_DIR "SOME_LOCAL_PATH") # replace this value with a real path
message("pybind11 dir: ${pybind11_DIR}")

find_package(pybind11 PATHS "${pybind11_DIR}")
