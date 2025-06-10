#include "transforms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(transforms, m) {
    m.doc() = "pybind11 example";

    py::class_<DCT_2<double>>(m, "dct_2")
        .def(py::init<int>())
        .def("transform", &DCT_2<double>::transform)
        .def("inverse", &DCT_2<double>::inverse);
}