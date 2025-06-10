#ifndef __TRANSFORMS_H__
#define __TRANSFORMS_H__

#include <vector>
#include <concepts>
#include <complex>
#include <ranges>



/**
 * Abstract type for fixed size invertible transforms that convert a floating point sequence
 * of up to length n to some domain specified by U
 */
template<std::floating_point T, typename U>
class Abstract_Transformer {
    public:
        Abstract_Transformer(unsigned int input_size) : input_size(input_size) {}
        const unsigned int input_size;

        /**
         *  Return 0 if successful, -1 otherwise
         */
        virtual int transform(const std::vector<T>& in, std::vector<U>& out) = 0;
        virtual int inverse(const std::vector<U>& in, std::vector<T>& out) = 0;
    
    protected:
        /**
         * Utility that finds the least multiple of 2 greater than n
         */
        static unsigned int radix_2_size(unsigned int n);

        /**
         * Utility function for transfering input indices to output indices
         */
        int bit_reversal(int bits, int num_bits);
};



template<std::floating_point T>
class FFT : public Abstract_Transformer<T, std::complex<T>> {
    private:
        int num_bits; // number of bits needed to represent an index of the output
        std::vector<std::complex<T>> W; // twiddle factors

    public:
        FFT(unsigned int input_size);        
        int transform(const std::vector<T>& in, std::vector<std::complex<T>>& out) override;
        int inverse(const std::vector<std::complex<T>>& in, std::vector<T>& out) override;
    
    private:
        /**
         *  Radix-2, decimation in time
         */
        void complex_fft(const std::vector<std::complex<T>>& in, std::vector<std::complex<T>>& out);
};

template<std::floating_point T>
class DCT_2 : public Abstract_Transformer<T, T> {
    private:
        FFT<T> fft_2n;
        std::vector<std::complex<T>> phase_factors;
        T coeff;
        const T beta_0 = 1/std::sqrt(2);
    public:
        DCT_2(unsigned int size);
        int transform(const std::vector<T>& in, std::vector<T>& out) override;
        int inverse(const std::vector<T>& in, std::vector<T>& out) override;
};

#include "abstract_transformer_impl.hpp"
#include "dct_impl.hpp"
#include "fft_impl.hpp"


#endif
