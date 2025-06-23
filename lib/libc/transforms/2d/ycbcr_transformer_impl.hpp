#ifndef __YCBCR_TRANSFORMER_IMPL_H__
#define __YCBCR_TRANSFORMER_IMPL_H__

#include "2d_transforms.hpp"

// TODO: is this verbosity okay? should it be chaned?
template<typename T>
YCbCr_Transformer<T>::YCbCr_Transformer(const Shape shape, float k_b, float k_r)
        : Abstract_Image_Transformer<T, T>(shape), k_b(k_b), k_r(k_r), k_g(1 - k_b - k_r),
        transform_matrix(std::span(this->forward_transform_data), 3, 3),
        inverse_matrix(std::span(this->inverse_transform_data), 3, 3) {
    
    // the matrices are views on the underlying data so can initialize data after creating the matrices
    // need to create the matrices first because static extent spans cannot point at nothing
    this->compute_forward_transform();
    this->compute_inverse_transform();
}


template<typename T>
int YCbCr_Transformer<T>::transform(Image_View<const T> in, Image_View<T> out) {
    // TODO: currently no enforcement on input and output both having the same element ordering or shape, is this the correct choice?
    if(in.size() > out.size()) {
        return -1;
    }

    float coeffs[] = {k_r, k_g, k_b}; // assuming RGB
    // first color channel sets the output values in case not zero initialized
    int k = 0;
    for(int i = 0; i < in.shape().m * in.shape().n; i ++) {
        out.index(k * out.strides().ch + i) = coeffs[k] * in.index(k * in.strides().ch + i);
    }
    // second and third color components are added
    for(k = 1; k < 3; k++) {
        for(int i = 0; i < in.shape().m * in.shape().n; i ++) {
            out.index(k * out.strides().ch + i) += in.index(k * in.strides().ch + i);
        }
    }
    return 0;
}

template<typename T>
void YCbCr_Transformer<T>::compute_forward_transform(){
    // First channel: Y
    this->forward_transform_data[0] = this->k_r;
    this->forward_transform_data[1] = this->k_g;
    this->forward_transform_data[2] = this->k_b;
    
    // Second channel: C_B
    this->forward_transform_data[3] = -0.5 * this->k_r/(1 - this->k_b);
    this->forward_transform_data[4] = -0.5 * this->k_g/(1 - this->k_b);
    this->forward_transform_data[5] = 0.5;

    // Third channel: C_R
    this->forward_transform_data[6] = 0.5;
    this->forward_transform_data[7] = -0.5 * this->k_g/(1 - this->k_r);
    this->forward_transform_data[8] = -0.5 * this->k_b/(1 - this->k_r);
}

template<typename T>
void YCbCr_Transformer<T>::compute_inverse_transform(){
    // First channel: Y
    this->inverse_transform_data[0] = 1;
    this->inverse_transform_data[1] = 0;
    this->inverse_transform_data[2] = 2 - 2*this->k_r;
    
    // Second channel: C_B
    this->inverse_transform_data[3] = 1;
    this->inverse_transform_data[4] = -this->k_b/this->k_g * (2 - 2*this->k_b);
    this->inverse_transform_data[5] = -this->k_r/this->k_g * (2 - 2*this->k_r);

    // Third channel: C_R
    this->inverse_transform_data[6] = 1;
    this->inverse_transform_data[7] = 2 - 2*this->k_b;
    this->inverse_transform_data[8] = 0;
}

#endif