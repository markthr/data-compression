#ifndef __FFT_IMPL_H__
#define __FFT_IMPL_H__

#include <numbers>
#include <iostream>
#include "transforms.hpp"



template<std::floating_point T>
FFT<T>::FFT(unsigned int input_size) : Abstract_Transformer<T, std::complex<T>>(FFT<T>::radix_2_size(input_size)){
    // set num_bit which is used for bit_reversal(...)
    uint size = this->input_size;
    this->num_bits = 0;
    while(size) {
        size >>= 1;
        this->num_bits++;
    }
    this->num_bits--; // only need bits for 2^n - 1;

    // initial twiddle factors for fft size
    W.resize(this->input_size);
    T coeff = -2 * std::numbers::pi / this->input_size;
    for(uint k = 0; k < this->input_size; k++) {
        W[k] = std::exp(std::complex<T>(0, coeff * k));
    }

    // std::cout << "Twiddle Factors:";
    // for(auto cnum: this->W) {
    //     std::cout << ' ' << cnum;
    // }
    // std::cout << '\n';
}

template<std::floating_point T>
int  FFT<T>::transform(const std::vector<T>& in, std::vector<std::complex<T>>& out){
    out.resize(this->input_size);
    this->complex_fft(std::vector<std::complex<T>>(in.begin(), in.end()), out);
    return 0;
}

template<std::floating_point T>
int  FFT<T>::inverse(const std::vector<std::complex<T>>& in, std::vector<T>& out){
    std::vector<std::complex<T>> c_in(this->input_size);
    out.resize(this->input_size);
    std::vector<std::complex<T>> c_out(this->input_size);
    T coeff = ((T) 1) / this->input_size;
    for(uint k = 0; k < this->input_size; k++) {
        c_in[k] = coeff * std::conj(in[k]);
    }
    
    this->complex_fft(c_in, c_out);

    for(uint k = 0; k < this->input_size; k++) {
        out[k] = std::real(c_out[k]);
    }

    return 0;
}



template<std::floating_point T>
void FFT<T>::complex_fft(const std::vector<std::complex<T>>& in, std::vector<std::complex<T>>& out) {
    // initial output with size 2 FFTs
    uint step_size = 2; // size of the FFT being computed
    uint offset = this->input_size/step_size; // offset between points input into 2-point FFTs

    for(uint k = 0; k < offset; k++) {
        uint l = k + offset;
        // std::cout << "Butterfly (" << this->bit_reversal(k) << ", " << this->bit_reversal(l)  <<
        //     ") for FFT of size: 2, W[0], step_size=" << step_size << '\n';

        std::complex<T> first = in[k]; // first index has to be in range
        std::complex<T> second = l < in.size() ? in[l] : 0; // zero pad end of input
        out[FFT<T>::bit_reversal(k, this->num_bits)] = first + second;
        out[FFT<T>::bit_reversal(l, this->num_bits)] = first - second;
    }

    // combine results
    // loop over powers of 2
    for(step_size = 4; step_size <= this->input_size; step_size *=2) {
        offset = step_size/2; // offset between points in butterflies that combine sub FFTs
        uint group_count = this->input_size/step_size;
        // loop over groups of transforms
        for(uint group_num = 0; group_num < group_count; group_num++) {
            // loop over each element in a group of transforms
            for(uint index = 0; index < offset; index++) {
                uint k = 2 * group_num * offset + index; // skip over group, each group is double the size of the offset
                // std::cout << "Butterfly (" << k << ", " << (k + offset) <<
                //     ") for FFT of size: " << step_size << " with W[" << (index*group_count) << "], group_count=" <<
                //     group_count << ", group_num=" << group_num << ", and index=" << index << "\n";
                
                std::complex<T> first = out[k];
                std::complex<T> second = out[k + offset] * this->W[index * group_count];

                out[k] = first + second;
                out[k + offset] = first - second;
            }
        }
    }
}

#endif