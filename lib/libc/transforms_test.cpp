#include <iostream>

#include "transforms.hpp"

#define N 8

int main() {
    FFT<float, N> fft_float;
    std::vector<float> input_float(N, 1);
    std::vector<std::complex<float>> output_cfloat(N);
    int res1 = fft_float.transform(input_float, output_cfloat);

    FFT<double, N> fft_double;
    std::vector<double> input_double(N, 0);
    input_double[2] = 1;
    std::vector<std::complex<double>> output_cdouble;
    int res2 = fft_double.transform(input_double, output_cdouble);

    // results printed in polar form
    std::cout << "Transforms completed with status1: " << res1
        << " and status2: " << res2 << '\n';
    
    std::cout << "fft_float result:";
    for(auto cnum: output_cfloat) {
        std::cout << '(' << std::abs(cnum) << ", " << std::arg(cnum) << ") ";
    }
    std::cout << '\n';
    
    std::cout << "fft_double result:";
    for(auto cnum: output_cdouble) {
        std::cout << '(' << std::abs(cnum) << ", " << std::arg(cnum) << ") ";
    }
    std::cout << '\n';
}