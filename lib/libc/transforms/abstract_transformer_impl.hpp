#ifndef __ABSTRACT_TRANFORMER_IMPL_H__
#define __ABSTRACT_TRANFORMER_IMPL_H__

#include "transforms.hpp"

template<std::floating_point T, typename U, template<typename> class Container>
Abstract_Transformer<T, U, Container>::Abstract_Transformer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {}

template<std::floating_point T, typename U, template<typename> class Container>
Abstract_Transformer<T, U, Container>::Abstract_Transformer(int size)
        : Abstract_Transformer(size, size) {}

template<std::floating_point T, typename U, template<typename> class Container>
std::vector<U> Abstract_Transformer<T, U, Container>::transform(Container<const T> in) {
    std::vector<U> result(this->output_size);
    this->transform(in, result);
    return result;
}

template<std::floating_point T, typename U, template<typename> class Container>
std::vector<T> Abstract_Transformer<T, U, Container>::inverse(Container<const U> in) {
    std::vector<T> result(this->input_size);
    this->inverse(in, result);
    return result;
}


template<std::floating_point T, typename U, template<typename> class Container>
int Abstract_Transformer<T, U, Container>::radix_2_size(int n) {
    // currently only support radix-2 FFT
    // increase FFT size to minimum power of 2 greater than or equal to the input size n
    if(n < 2) {
        // TODO: determine a satisfying way to reject values of n that are less than 1.
        n = 2;
    }
    else {
        // over optimized way to find the smallest power of 2 greater than n
        // works by making all bits smaller than the MSB 1 which yields a number
        // of the form 2^n - 1, add 1 and get a power of 2.
        n = n - 1;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n++;
    }

    return n;
}

template<std::floating_point T, typename U, template<typename> class Container>
int Abstract_Transformer<T, U, Container>::bit_reversal(int bits, int num_bits) {
    int reversed = 0;
    for(int k = 0; k < num_bits; k++) {
        reversed = (reversed << 1) + (bits & 1); // shift in each LSB to output
        bits >>= 1; // shift out each LSB from input
    }
    return reversed;
}

#endif
