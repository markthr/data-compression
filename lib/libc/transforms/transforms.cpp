#include "transforms.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Py_DCT : public DCT_2<double> {
    public:
        using DCT_2<double>::DCT_2;

        std::vector<double> py_transform(py::array_t<double, py::array::c_style>& in) {
            return this->transform(std::span(in.data(), in.size()));
        }

        std::vector<double> py_inverse(py::array_t<double, py::array::c_style>& in) {
            return this->inverse(std::span(in.data(), in.size()));
        }
};

class Py_FFT : public FFT<double> {
    public:
        using FFT<double>::FFT;

        std::vector<std::complex<double>> py_transform(py::array_t<double, py::array::c_style>& in) {
            return this->transform(std::span(in.data(), in.size()));
        }

        std::vector<double> py_inverse(py::array_t<std::complex<double>, py::array::c_style>& in) {
            return this->inverse(std::span(in.data(), in.size()));
        }
};


PYBIND11_MODULE(transforms, m) {
    m.doc() = "pybind11 example";
    
    py::class_<Py_DCT>(m, "dct_2")
        .def(py::init<int>())
        .def("transform",
            &Py_DCT::py_transform)
        .def("inverse",
            &Py_DCT::py_inverse);
    
    py::class_<Py_FFT>(m, "fft")
        .def(py::init<int>())
        .def("transform",
            &Py_FFT::py_transform)
        .def("inverse",
            &Py_FFT::py_inverse);
}