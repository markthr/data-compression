#include "transforms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(transforms, m) {
    m.doc() = "pybind11 example";

    py::class_<DCT_2<double, 8>>(m, "dct_8")
        .def(py::init<>())
        .def("transform", &DCT_2<double, 8>::transform)
        .def("inverse", &DCT_2<double, 8>::inverse);
}