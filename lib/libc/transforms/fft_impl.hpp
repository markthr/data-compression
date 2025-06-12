#ifndef __FFT_IMPL_H__
#define __FFT_IMPL_H__

#include <numbers>
#include <iostream>
#include <ranges>
#include "transforms.hpp"




template<std::floating_point T>
FFT<T>::FFT(int size)
            : Abstract_Transformer<T, std::complex<T>>(Abstract_Transformer<T, std::complex<T>>::radix_2_size(size)){
    
    // set num_bit which is used for bit_reversal(...)
    this->num_bits = 0;
    int shifted_size = this->input_size;
    while(shifted_size) {
        shifted_size >>= 1;
        this->num_bits++;
    }
    this->num_bits--; // only need bits for 2^n - 1;

    // initial twiddle factors for fft size
    W.resize(this->input_size);
    T coeff = -2 * std::numbers::pi / this->input_size;
    for(int k = 0; k < this->input_size; k++) {
        W[k] = std::exp(std::complex<T>(0, coeff * k));
    }

    // std::cout << "Twiddle Factors:";
    // for(auto cnum: this->W) {
    //     std::cout << ' ' << cnum;
    // }
    // std::cout << '\n';
}

template<std::floating_point T>
int  FFT<T>::transform(std::span<const T> in, std::span<std::complex<T>> out){
    // TODO: could enforce size requirement at compile time, worth considering
    if(std::ranges::ssize(out) < this->input_size) {
        return -1;
    }

    // get a view of the input as if it were complex
    auto r = std::views::transform(in,
        [](T val) {
            return std::complex<T>(val, 0);
        }
    );
    std::vector<std::complex<T>> c_in(r.begin(), r.end());

    this->complex_fft(c_in, out);
    return 0;
}

template<std::floating_point T>
int  FFT<T>::inverse(std::span<const std::complex<T>> in, std::span<T> out){
    // TODO: could enforce size requirement at compile time, worth considering
    if(std::ranges::ssize(out) < this->input_size) {
        return -1;
    }

    // TODO: look into reducing the amount of copies made here
    // could have a buffer (or multiple) that are held by the transformer throughout its
    // lifetime i.e. RAII. 
    std::vector<std::complex<T>> c_in(this->input_size);
    std::vector<std::complex<T>> c_out(this->input_size);
    T coeff = ((T) 1) / this->input_size;
    for(int k = 0; k < this->input_size; k++) {
        c_in[k] = coeff * std::conj(in[k]);
    }
    
    this->complex_fft(c_in, c_out);

    for(int k = 0; k < this->input_size; k++) {
        out[k] = std::real(c_out[k]);
    }

    return 0;
}



template<std::floating_point T>
void FFT<T>::complex_fft(std::span<const std::complex<T>> in, std::span<std::complex<T>> out) {
    // initial output with size 2 FFTs
    int step_size = 2; // size of the FFT being computed
    int offset = this->input_size/step_size; // offset between points input into 2-point FFTs

    // perform initial size 2 FFTs
    for(int k = 0; k < offset; k++) {
        int l = k + offset;
        // std::cout << "Butterfly (" << this->bit_reversal(k) << ", " << this->bit_reversal(l)  <<
        //     ") for FFT of size: 2, W[0], step_size=" << step_size << '\n';

        std::complex<T> first = in[k]; // first index has to be in range
        std::complex<T> second = l < std::ranges::ssize(in) ? in[l] : 0; // zero pad end of input
        out[FFT<T>::bit_reversal(k, this->num_bits)] = first + second;
        out[FFT<T>::bit_reversal(l, this->num_bits)] = first - second;
    }

    // combine results
    // loop over powers of 2
    for(step_size = 4; step_size <= this->input_size; step_size *=2) {
        offset = step_size/2; // offset between points in butterflies that combine sub FFTs
        int group_count = this->input_size/step_size;
        // loop over groups of transforms
        for(int group_num = 0; group_num < group_count; group_num++) {
            // loop over each element in a group of transforms
            for(int index = 0; index < offset; index++) {
                int k = 2 * group_num * offset + index; // skip over group, each group is double the size of the offset
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