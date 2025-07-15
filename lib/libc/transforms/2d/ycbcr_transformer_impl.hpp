#ifndef __YCBCR_TRANSFORMER_IMPL_H__
#define __YCBCR_TRANSFORMER_IMPL_H__

#include "2d_transforms.hpp"

#include <vector>

// TODO: is this verbosity okay? should it be chaned?
template<typename T>
YCbCr_Transformer<T>::YCbCr_Transformer(const Shape shape, float k_b, float k_r)
        : Abstract_Matrix_Transformer<T, T, Channels>(shape), k_b(k_b), k_r(k_r), k_g(1 - k_b - k_r),
        transform_matrix({3, 3}), inverse_matrix({3, 3}) {
    
    // initialize values in the transform/inverse matrices
    this->compute_forward_transform();
    this->compute_inverse_transform();
}

template<typename T>
int YCbCr_Transformer<T>::transform_impl(const Image_Matrix<T, const_vector_t>& in, Image_Matrix<T> out) {
    // TODO: currently no enforcement on input and output both having the same element ordering or shape, is this the correct choice?
    if(in.size() > out.size()) {
        return -1;
    }


    return mat::transform_channels(in, this->transform_matrix, out);
}

template<typename T>
int YCbCr_Transformer<T>::inverse_impl(const Image_Matrix<T, const_vector_t>& in, Image_Matrix<T> out) {
    // TODO: currently no enforcement on input and output both having the same element ordering or shape, is this the correct choice?
    if(in.size() > out.size()) {
        return -1;
    }

    return mat::transform_channels(in, this->inverse_matrix, out);
}

template<typename T>
void YCbCr_Transformer<T>::compute_forward_transform(){
    // First channel: Y
    this->transform_matrix.index(0, 0) = this->k_r;
    this->transform_matrix.index(0, 1) = this->k_g;
    this->transform_matrix.index(0, 2) = this->k_b;
    
    // Second channel: C_B
    this->transform_matrix.index(1, 0) = -0.5 * this->k_r/(1 - this->k_b);
    this->transform_matrix.index(1, 1) = -0.5 * this->k_g/(1 - this->k_b);
    this->transform_matrix.index(1, 2) = 0.5;

    // Third channel: C_R
    this->transform_matrix.index(2, 0) = 0.5;
    this->transform_matrix.index(2, 1) = -0.5 * this->k_g/(1 - this->k_r);
    this->transform_matrix.index(2, 2) = -0.5 * this->k_b/(1 - this->k_r);
}

template<typename T>
void YCbCr_Transformer<T>::compute_inverse_transform(){
    // First channel: Y
    this->inverse_matrix.index(0, 0) = 1;
    this->inverse_matrix.index(0, 1) = 0;
    this->inverse_matrix.index(0, 2) = 2 - 2*this->k_r;
    
    // Second channel: C_B
    this->inverse_matrix.index(1, 0) = 1;
    this->inverse_matrix.index(1, 1) = -this->k_b/this->k_g * (2 - 2*this->k_b);
    this->inverse_matrix.index(1, 2) = -this->k_r/this->k_g * (2 - 2*this->k_r);

    // Third channel: C_R
    this->inverse_matrix.index(2, 0) = 1;
    this->inverse_matrix.index(2, 1) = 2 - 2*this->k_b;
    this->inverse_matrix.index(2, 2) = 0;
}

#endif