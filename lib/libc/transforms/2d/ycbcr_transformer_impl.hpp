#ifndef __YCBCR_TRANSFORMER_IMPL_H__
#define __YCBCR_TRANSFORMER_IMPL_H__

#include "2d_transforms.hpp"

// TODO: is this verbosity okay? should it be chaned?
template<typename T>
YCbCr_Transformer<T>::YCbCr_Transformer(const Shape shape, float k_b, float k_r)
        : Abstract_Image_Transformer<T, T>(shape), k_b(k_b), k_r(k_r), k_g(1 - k_b - k_r),
        forward_transform_data(get_forward_transform(k_r, 1 - k_b - k_r, k_b)){
    
    std::span<T> dspan(this->forward_transform_data.data(), 9);
    this->transform_matrix = Multichannel_Matrix<T, 1>(dspan, 3, 3);
    // code go here 
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
std::array<T, 9> YCbCr_Transformer<T>::get_forward_transform(float k_r, float k_g, float k_b){
    // First channel: Y
    std::array<T, 9> forward_transform = {k_r, k_g, k_b};
    
    // Second channel: C_B
    forward_transform[3] = -0.5 * k_r/(1 - k_b);
    forward_transform[4] = -0.5 * k_g/(1 - k_b);
    forward_transform[5] = 0.5;

    // Third channel: C_R
    forward_transform[6] = 0.5;
    forward_transform[7] = -0.5 * k_g/(1 - k_r);
    forward_transform[8] = -0.5 * k_b/(1 - k_r);

    return forward_transform;
}

#endif