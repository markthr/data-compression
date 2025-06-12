#ifndef __DCT_IMPL_H__
#define __DCT_IMPL_H__

#include "transforms.hpp"
#include <ranges>

template<std::floating_point T>
DCT_2<T>::DCT_2(int size)
        : Abstract_Transformer<T, T>(DCT_2<T>::radix_2_size(size)), fft_2n(FFT<T>(this->input_size*2)) {
    
    this->coeff = 1/std::sqrt(this->fft_2n.input_size); // fft size is twice as big as N in common DCT definition

    int output_size = this->fft_2n.input_size/2;
    this->phase_factors.resize(output_size);
    T phase_coeff = -std::numbers::pi / this->fft_2n.input_size;
    
    for(int k = 0; k < output_size; k++) {
        this->phase_factors[k] = std::exp(std::complex<T>(0, phase_coeff * k));
    }
}

//TODO: this does not take advantage of being able to compute N point DCTs with only an N point FFT when N is an even number
template<std::floating_point T>
int  DCT_2<T>::transform(std::span<const T> in, std::span<T> out){
    // TODO: could enforce size requirement at compile time, worth considering
    if(std::ranges::ssize(out) < this->input_size) {
        return -1;
    }

    // TODO: look into reducing the amount of copies made here, see notes on FFT
    std::vector<T> in_extended(this->fft_2n.input_size);

    int n = this->input_size < std::ranges::ssize(in) ? this->input_size : std::ranges::ssize(in);
    // construct half symmetric period extension with zero padding to make size a factor of 2
    for(int k = 0; k < n; k++) {
        in_extended[k] = in[k];
        in_extended[this->fft_2n.input_size - k - 1] = in[k];
    }
    

    std::vector<std::complex<T>> out_extended(this->fft_2n.input_size);
    int status = this->fft_2n.transform(in_extended, out_extended);
    if(status) {
        return status; // indicates an error
    }


    // TODO: think about handling empty inputs, currently assumed output has at least 1 element
    out[0] = this->coeff * this-> beta_0 * std::real(out_extended[0]);
    for(int k = 1; k < this->input_size; k++) {
        out[k] = this->coeff * std::real(this->phase_factors[k] * out_extended[k]);
    }

    return 0;
}

template<std::floating_point T>
int  DCT_2<T>::inverse(std::span<const T> in, std::span<T> out){
    // TODO: could enforce size requirement at compile time, worth considering
    if(std::ranges::ssize(out) < this->input_size) {
        return -1;
    }
    
    // TODO: look into reducing the amount of copies made here, see notes on FFT
    std::vector<std::complex<T>> in_extended(this->fft_2n.input_size);

    // construct DFT for half symmetric period extension from DCT values
    in_extended[0] = in[0]/(this->coeff * this->beta_0);

    T coeff_i = 1/this->coeff;
    for(int k = 1; k < this->input_size; k ++ ) {
        in_extended[k] = coeff_i * std::conj(this->phase_factors[k]) * in[k];
    }

    in_extended[this->input_size] = 0;

    for(int k = 1; k < this->input_size; k++) {
        in_extended[this->input_size + k] = std::complex<T>(0, -coeff_i) * std::conj(this->phase_factors[k]) * in[this->input_size - k];
    }

    std::vector<T> out_extended(this->fft_2n.input_size);
    int status = this->fft_2n.inverse(in_extended, out_extended);
    if(status) {
        return status;
    }

    for(int k = 0; k < this->input_size; k++) {
        out[k] = out_extended[k];
    }
    return 0;
}


#endif