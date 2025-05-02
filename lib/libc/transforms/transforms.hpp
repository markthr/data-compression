#ifndef __TRANSFORMS_H__
#define __TRANSFORMS_H__

#include <vector>
#include <concepts>
#include <complex>

/**
 * Abstract type for invertible transforms that convert a floating point sequence
 * of length n to some domain specified by U
 */
template<std::floating_point T, typename U, int n>
class Abstract_Transformer {
    public:
        /**
         *  Return 0 if successful, -1 otherwise
         */
        virtual int transform(const std::vector<T>& in, std::vector<U>& out) = 0;
        virtual int inverse(const std::vector<U>& in, std::vector<T>& out) = 0;
};

template<std::floating_point T, int n>
class DCT_2 : Abstract_Transformer<T, T, n> {
    public:
        int transform(const std::vector<T>& in, std::vector<T>& out) override;
        int inverse(const std::vector<T>& in, std::vector<T>& out) override;
};

template<std::floating_point T, int n>
class FFT : Abstract_Transformer<T, std::complex<T>, n> {
    private:
        int fft_size; // the least multiple of 2 greater than n
        int num_bits; // number of bits needed to represent an index of the output
        std::vector<std::complex<T>> W; // twiddle factors

    public:
        FFT();
        int transform(const std::vector<T>& in, std::vector<std::complex<T>>& out) override;
        int inverse(const std::vector<std::complex<T>>& in, std::vector<T>& out) override;
    private:
        /**
         *  Radix-2, decimation in time
         */
        void fft_step(int offset, int scale, const std::vector<T>& in, std::vector<std::complex<T>>& out);
        /**
         * Utility function for transfering input indices to output indices
         */
        int bit_reversal(int index);
};

#include "dct_impl.hpp"
#include "fft_impl.hpp"


#endif