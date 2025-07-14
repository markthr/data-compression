#ifndef __2D_TRANSFORMS_H__
#define __2D_TRANSFORMS_H__

#include <span>
#include <array>
#include <memory>
#include <utility>
#include <initializer_list>
#include "../transforms.hpp"
#include "multichannel_matrix_impl.hpp"

/**
 * Abstract type for fixed size invertible transforms that convert a floating point sequence
 * of up to length n to some domain specified by U
 */
template<std::floating_point T, typename U, template<typename> class Container = std::span>
class Abstract_Transformer_NC {
    public:
        const std::size_t input_size;
        const std::size_t output_size;

        Abstract_Transformer_NC(std::size_t input_size, std::size_t output_size) : input_size(input_size), output_size(output_size) {}

        Abstract_Transformer_NC(std::size_t size) : Abstract_Transformer_NC(size, size) {}

        /**
         *  Return 0 if successful, -1 otherwise
         */
        virtual int transform(Container<T> in, Container<U> out) = 0;
        virtual int inverse(Container<U> in, Container<T> out) = 0;
        

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

// template<typename T>
// typedef shared_vec<T> std::shared_ptr<std::vector<T>>



// needs to be declared before ycbcr transformer
#include "matrix_operations_impl.hpp"

template<typename T, typename U>
class Abstract_Image_Transformer : public Abstract_Transformer_NC<T, U, Image_Matrix> {
    
    public:
        const Shape input_shape;
        const Shape output_shape;

        Abstract_Image_Transformer(const Shape shape)
            : input_shape(shape), output_shape(shape), 
            Abstract_Transformer_NC<T, U, Image_Matrix>(
                shape.m * shape.n * 3) {}
        
        Abstract_Image_Transformer(const Shape input_shape, const Shape output_shape)
            : input_shape(input_shape), output_shape(output_shape),
            Abstract_Transformer_NC<T, U, Image_Matrix>(
                input_shape.m * input_shape.n * 3, output_shape.m * output_shape.n * 3) {}

};







template<typename T>
class YCbCr_Transformer : public Abstract_Image_Transformer<T, T> {
    public:
        static const int Channels = 3;
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
        
        // indicate override of pure virtual signature
        int transform(Image_Matrix<T> in, Image_Matrix<T> out) override;
        int inverse(Image_Matrix<T> in, Image_Matrix<T> out) override;

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