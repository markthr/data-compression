#ifndef __FFT_IMPL_H__
#define __FFT_IMPL_H__

#include <numbers>
#include <iostream>
#include "transforms.hpp"

template<std::floating_point T, int n>
FFT<T, n>::FFT() {
    // increase FFT size to minimum power of 2 greater than or equal to the input size n
    if(n < 1) {
        // TODO: determine a satisfying way to reject values of n that are less than 1.
        this->fft_size = 2;
    }
    else {
        // over optimized way to find the smallest power of 2 greater than n
        // works by making all bits smaller than the MSB 1 which yields a number
        // of the form 2^n - 1, add 1 and get a power of 2.
        this->fft_size = n - 1;
        this->fft_size |= this->fft_size >> 1;
        this->fft_size |= this->fft_size >> 2;
        this->fft_size |= this->fft_size >> 4;
        this->fft_size |= this->fft_size >> 8;
        this->fft_size |= this->fft_size >> 16;
        this->fft_size++;
    }

    // set num_bit which is used for bit_reversal(...)
    int size = this->fft_size;
    this->num_bits = 0;
    while(size) {
        size >>= 1;
        this->num_bits++;
    }
    this->num_bits--; // only need bits for 2^n - 1;



    // initial twiddle factors for fft size
    W.resize(this->fft_size);
    T coeff = -2 * std::numbers::pi / this->fft_size;
    for(int k = 0; k < this->fft_size; k++) {
        W[k] = std::exp(std::complex<T>(0, coeff * k));
    }

    // std::cout << "Twiddle Factors:";
    // for(auto cnum: this->W) {
    //     std::cout << ' ' << cnum;
    // }
    // std::cout << '\n';
}

template<std::floating_point T, int n>
int  FFT<T, n>::transform(const std::vector<T>& in, std::vector<std::complex<T>>& out){
    out.resize(this->fft_size);
    this->complex_fft(std::vector<std::complex<T>>(in.begin(), in.end()), out);
    return 0;
}

template<std::floating_point T, int n>
int  FFT<T, n>::inverse(const std::vector<std::complex<T>>& in, std::vector<T>& out){
    std::vector<std::complex<T>> c_in(this->fft_size);
    out.resize(n);
    std::vector<std::complex<T>> c_out(this->fft_size);
    T coeff = ((T) 1) / this->fft_size;
    for(int k = 0; k < this->fft_size; k++) {
        c_in[k] = coeff * std::conj(in[k]);
    }
    
    this->complex_fft(c_in, c_out);

    for(int k = 0; k < n; k++) {
        out[k] = std::real(c_out[k]);
    }

    return 0;
}

template<std::floating_point T, int n>
int  FFT<T, n>::bit_reversal(int index) {
    int reversed = 0;
    for(int k = 0; k < this->num_bits; k++) {
        reversed = (reversed << 1) + (index & 1); // shift in each LSB to output
        index >>= 1; // shift out each LSB from input
    }
    return reversed;
}

template<std::floating_point T, int n>
void FFT<T, n>::complex_fft(const std::vector<std::complex<T>>& in, std::vector<std::complex<T>>& out) {
    // initial output with size 2 FFTs
    int step_size = 2; // size of the FFT being computed
    int offset = this->fft_size/step_size; // offset between points input into 2-point FFTs

    for(int k = 0; k < offset; k++) {
        int l = k + offset;
        // std::cout << "Butterfly (" << this->bit_reversal(k) << ", " << this->bit_reversal(l)  <<
        //     ") for FFT of size: 2, W[0], step_size=" << step_size << '\n';

        std::complex<T> first = in[k]; // first index has to be in range
        std::complex<T> second = l < std::ssize(in) ? in[l] : 0; // zero pad end of input
        out[this->bit_reversal(k)] = first + second;
        out[this->bit_reversal(l)] = first - second;
    }

    // combine results
    // loop over powers of 2
    for(step_size = 4; step_size <= this->fft_size; step_size *=2) {
        offset = step_size/2; // offset between points in butterflies that combine sub FFTs
        int group_count = this->fft_size/step_size;
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

template<std::floating_point T, int n>
int  FFT<T, n>::size() {
    return this->fft_size;
}

#endif