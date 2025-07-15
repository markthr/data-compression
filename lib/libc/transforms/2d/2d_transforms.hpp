#ifndef __2D_TRANSFORMS_H__
#define __2D_TRANSFORMS_H__

#include <span>
#include <array>
#include <memory>
#include <utility>
#include <initializer_list>
#include <type_traits>

#include "../transforms.hpp"
#include "multichannel_matrix.hpp"



// template<typename T>
// typedef shared_vec<T> std::shared_ptr<std::vector<T>>



// needs to be declared before ycbcr transformer
#include "matrix_operations.hpp"

template<Has_Arithmetic T, Has_Arithmetic U, int Channels, std::size_t Extent = std::dynamic_extent>
class Abstract_Matrix_Transformer {
private:
    virtual int transform_impl(Multichannel_Matrix<T, Channels, Extent, const_vector_t> in, Multichannel_Matrix<U, Channels, Extent> out) = 0;
    virtual int inverse_impl(Multichannel_Matrix<U, Channels, Extent, const_vector_t> in, Multichannel_Matrix<T, Channels, Extent> out) = 0;
public:
    const Shape input_shape;
    const Shape output_shape;
    
    const std::size_t input_size;
    const std::size_t output_size;
    

    Abstract_Matrix_Transformer(const Shape input_shape, const Shape output_shape)
            : input_shape(input_shape), input_size(input_shape.m * input_shape.n * 3),
            output_shape(output_shape), output_size(output_shape.m * output_shape.n * 3) {}
    
    Abstract_Matrix_Transformer(const Shape shape)
            : Abstract_Matrix_Transformer<T, U, Channels, Extent>(shape, shape) {}
    
    


    /**
     *  Return 0 if successful, -1 otherwise
     */
    template<Matrix_Like<T> M_T>
    int transform(M_T& in, Multichannel_Matrix<U, Channels, Extent> out){
        Multichannel_Matrix<T, Channels, Extent, const_vector_t> in_mat = Multichannel_Matrix<T, Channels, Extent, const_vector_t>::as_matrix(in);

        return transform_impl(in_mat, out);
    }
    template<Matrix_Like<U> M_U>
    int inverse(M_U& in, Multichannel_Matrix<T, Channels, Extent> out) {
        Multichannel_Matrix<U, Channels, Extent, const_vector_t> in_mat = Multichannel_Matrix<U, Channels, Extent, const_vector_t>::as_matrix(in);

        return inverse_impl(in_mat, out);
    }

private:
    // 

};







template<Has_Arithmetic T>
class YCbCr_Transformer : public Abstract_Matrix_Transformer<T, T, 3> {
public:
    static const int Channels = 3;
private:
    // indicate override of pure virtual signature
    int transform_impl(Image_Matrix<T, const_vector_t> in, Image_Matrix<T> out) override;
    int inverse_impl(Image_Matrix<T, const_vector_t> in, Image_Matrix<T> out) override;
public:
    const float k_r;
    const float k_g;
    const float k_b;
    Multichannel_Matrix<T, 1> transform_matrix;
    Multichannel_Matrix<T, 1> inverse_matrix;

    /**
     * Default values for k_b and k_r are taken from ITU-R BT.601
     * 
     * k_g is determined by the formula k_b + k_r + k_g = 1
     * 
     * There are thus ways that bad values can be provided e.g. making one of the coefficients zero
     */
    YCbCr_Transformer(const Shape shape, float k_b = 0.114, float k_r = 0.299);
    // TODO: decide how to handle bad parameters in the constructor e.g. exception, public method which wraps a private constructor, etc
    
    

    /**
     * Use the current values for k_r, k_g, and k_b to compute and set forward transform matrix
     */
    void compute_forward_transform();

    /**
     * Use the current values for k_r, k_g, and k_b to compute and set forward inverse matrix
     */
    void compute_inverse_transform();
};


#include "ycbcr_transformer_impl.hpp"
#include "bitdepth_transformer_impl.hpp"

#endif