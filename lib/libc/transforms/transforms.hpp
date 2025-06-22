#ifndef __TRANSFORMS_H__
#define __TRANSFORMS_H__

#include <vector>
#include <concepts>
#include <complex>
#include <span>


/**
 * Abstract type for fixed size invertible transforms that convert a floating point sequence
 * of up to length n to some domain specified by U
 */
template<std::floating_point T, typename U, template<typename> class Container = std::span>
class Abstract_Transformer {
    public:
        Abstract_Transformer(int input_size, int output_size);
        Abstract_Transformer(int size);
        const int input_size;
        const int output_size;

        /**
         *  Return 0 if successful, -1 otherwise
         */
        virtual int transform(Container<const T> in, Container<U> out) = 0;
        virtual int inverse(Container<const U> in, Container<T> out) = 0;

        std::vector<U> transform(Container<const T> in);
        std::vector<T> inverse(Container<const U> in);
    
    // helper methods, a new copy is made for every parameterization of the template
    // perhaps it could eventually be worthwhile to make static versions that exist outside of templates
    protected:
        /**
         * Utility that finds the least multiple of 2 greater than n
         */
        static int radix_2_size(int n);

        /**
         * Utility function for transfering input indices to output indices
         */
        static int bit_reversal(int bits, int num_bits);
};


template<std::floating_point T>
class FFT : public Abstract_Transformer<T, std::complex<T>> {
    private:
        int num_bits; // number of bits needed to represent an index of the output
        std::vector<std::complex<T>> W; // twiddle factors

    public:
        FFT(int size);
        // allow overloading of the base class methods
        using Abstract_Transformer<T, std::complex<T>>::transform;
        using Abstract_Transformer<T, std::complex<T>>::inverse;
        // indicate override of pure virtual signature
        int transform(std::span<const T> in, std::span<std::complex<T>> out) override;
        int inverse(std::span<const std::complex<T>> in, std::span<T> out) override;
    
    private:
        /**
         * Radix-2, decimation in time
         * 
         * This method assumes that in and out have matching size that is equal to a power of 2
         * No return method as this method is designed to be performed after all failure checks have already happened
         */
        void complex_fft(std::span<const std::complex<T>> in, std::span<std::complex<T>> out);
};

template<std::floating_point T>
class DCT_2 : public Abstract_Transformer<T, T> {
    private:
        FFT<T> fft_2n;
        std::vector<std::complex<T>> phase_factors;
        T coeff;
        const T beta_0 = 1/std::sqrt(2);
    public:
        DCT_2(int size);
        // allow overloading of the base class methods
        using Abstract_Transformer<T, T>::transform;
        using Abstract_Transformer<T, T>::inverse;
        // indicate override of pure virtual signature
        int transform(std::span<const T> in, std::span<T> out) override;
        int inverse(std::span<const T> in, std::span<T> out) override;
};

#include "abstract_transformer_impl.hpp"
#include "dct_impl.hpp"
#include "fft_impl.hpp"


#endif
