#include<iostream>

#include "transforms.hpp"

#define N 8

int main() {
    DCT_2<float, N> dct2_float;
    std::vector<float> input_float(N);
    std::vector<float> output_float(N);
    int res1 = dct2_float.transform(input_float, output_float);

    DCT_2<double, N> dct2_double;
    std::vector<double> input_double(N);
    std::vector<double> output_double(N);
    int res2 = dct2_double.transform(input_double, output_double);

    std:: cout << "Transforms completed with status1: " << res1
        << " and status2: " << res2 << '\n';
}