#include "transforms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(transforms, m) {
    m.doc() = "pybind11 example";

    //TODO: pybind11 does not understand how to convert python arrays to spans, current code is a placeholder
    // there are forks of pybind11 that give span a default converter along with python libraries that port
    // C++ spans into python. Relying on more libraries seems not worthwhile in this case though.
    py::class_<DCT_2<double>>(m, "dct_2")
        .def(py::init<int>())
        .def("transform",
            static_cast<std::vector<double> (DCT_2<double>::*) (std::span<const double>)>(
                &DCT_2<double>::transform)
        )
        .def("inverse",
            static_cast<std::vector<double> (DCT_2<double>::*) (std::span<const double>)>(
                &DCT_2<double>::inverse)
        );
}