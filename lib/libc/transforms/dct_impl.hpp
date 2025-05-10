#ifndef __DCT_IMPL_H__
#define __DCT_IMPL_H__

#include "transforms.hpp"

template<std::floating_point T, int n>
DCT_2<T, n>::DCT_2() {
    this->coeff = 1/std::sqrt(this->fft_2n.size()); // fft size is twice as big as N in common DCT definition

    int output_size = this->fft_2n.size()/2;
    this->phase_factors.resize(output_size);
    T phase_coeff = -std::numbers::pi / this->fft_2n.size();
    
    for(int k = 0; k < output_size; k++) {
        this->phase_factors[k] = std::exp(std::complex<T>(0, phase_coeff * k));
    }
}

//TODO: this does not take advantage of being able to compute N point DCTs with only an N point FFT when N is an even number
template<std::floating_point T, int n>
int  DCT_2<T, n>::transform(const std::vector<T>& in, std::vector<T>& out){
    int size = 2 * n;
    std::vector<T> in_extended(size);

    // construct half symmetric period extension
    for(int k = 0; k < n; k++) {
        in_extended[k] = in[k];
        in_extended[size - k - 1] = in[k];
    }
    

    
    std::vector<std::complex<T>> out_extended(this->fft_2n.size());
    int status = this->fft_2n.transform(in_extended, out_extended);
    if(status) {
        return status;
    }

    int output_size = this->fft_2n.size()/2;
    out.resize(output_size);

    // TODO: think about handling empty inputs, currently assumed output has at least 1 element
    out[0] = this->coeff * this-> beta_0 * std::real(out_extended[0]);
    for(int k = 1; k < output_size; k++) {
        out[k] = this->coeff * std::real(this->phase_factors[k] * out_extended[k]);
    }

    return 0;
}

template<std::floating_point T, int n>
int  DCT_2<T, n>::inverse(const std::vector<T>& in, std::vector<T>& out){
    int size = this->fft_2n.size()/2;
    std::vector<std::complex<T>> in_extended(this->fft_2n.size());

    // construct DFT for half symmetric period extension from DCT values
    in_extended[0] = in[0]/(this->coeff * this->beta_0);

    T coeff_i = 1/this->coeff;
    for(int k = 1; k < size; k ++ ) {
        in_extended[k] = coeff_i * std::conj(this->phase_factors[k]) * in[k];
    }

    in_extended[size] = 0;

    for(int k = 1; k < size; k++) {
        in_extended[size + k] = std::complex<T>(0, -coeff_i) * std::conj(this->phase_factors[k]) * in[size - k];
    }

    std::vector<T> out_extended(2 * n);
    int status = this->fft_2n.inverse(in_extended, out_extended);
    if(status) {
        return status;
    }

    out.resize(n);
    for(int k = 0; k < n; k++) {
        out[k] = out_extended[k];
    }
    return 0;
}


#endif