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


PYBIND11_MODULE(transforms, m) {
    m.doc() = "pybind11 example";

    //TODO: pybind11 does not understand how to convert python arrays to spans, current code is a placeholder
    // there are forks of pybind11 that give span a default converter along with python libraries that port
    // C++ spans into python. Relying on more libraries seems not worthwhile in this case though.

    /**
     * code pasted into the terminal for testing
    python - << 'EOF'
    import numpy as np
    import transforms as tm
    dct = tm.dct_2(8)
    res = dct.transform(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    print(res)
    EOF
     *
    Exptected Results: [12.727922061357855, -6.442323022705137, 0.0, -0.6734548009039405, 0.0, -0.20090290373599626, 0.0, -0.05070232275964579]
     */
    
    py::class_<Py_DCT>(m, "dct_2")
        .def(py::init<int>())
        .def("transform",
            &Py_DCT::py_transform)
        .def("inverse",
            &Py_DCT::py_inverse);
}