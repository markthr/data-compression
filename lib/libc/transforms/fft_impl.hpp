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
    this->fft_step(0, 1, std::vector<std::complex<T>>(in.begin(), in.end()), out);
    return 0;
}

template<std::floating_point T, int n>
int  FFT<T, n>::inverse(const std::vector<std::complex<T>>& in, std::vector<T>& out){
    std::vector<std::complex<T>> c_in(in.size());
    out.resize(n);
    std::vector<std::complex<T>> c_out(n);
    T coeff = ((T) 1) / in.size();
    for(int k = 0; k < std::ranges::ssize(in); k++) {
        c_in[k] = coeff * std::conj(in[k]);
    }
    
    this->fft_step(0, 1, c_in, c_out);

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
void FFT<T, n>::fft_step(int offset, int scale, const std::vector<std::complex<T>>& in, std::vector<std::complex<T>>& out) {
    if(scale * 2 >= this->fft_size) { // size 2 FFT
        std::complex<T> first = in[offset]; // first index has to be in range
        std::complex<T> second = offset + scale < n ? in[offset + scale] : 0; // zero pad end of input
        out[this->bit_reversal(offset)] = first + second;
        out[this->bit_reversal(offset + scale)] = first - second;
    }
    else {
        this->fft_step(offset, scale * 2, in, out); // FFT of even indices
        this->fft_step(offset + scale, scale * 2, in, out); // FFT of odd indices

        // combine the results
        int step_size = this->fft_size/scale; // the size of the FFT being computed in this recursion
        for(int k = 0; k < step_size/2; k++) {
            int index = offset * step_size + k;
            std::complex<T> first = out[index];
            std::complex<T> second = out[index + step_size/2] * this->W[k * scale];
            out[index] = first + second;
            out[index + step_size/2] = first - second;

            // std::cout << "Butterfly of X_" << index << " and X_" << (index + step_size/2)
            //     << " for FFT of size: " << step_size << " and W[" << (k*scale) << "]\n";
        }
    }
}

#endif