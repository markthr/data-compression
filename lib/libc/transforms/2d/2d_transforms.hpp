#ifndef __2D_TRANSFORMS_H__
#define __2D_TRANSFORMS_H__

#include <span>
#include <array>
#include "../transforms.hpp"

struct Shape {
    int m; // m=height=# of rows
    int n; // n=width=# of cols
};

template<typename T, int Channels, std::size_t Extent = std::dynamic_extent>
class Multichannel_Matrix {
    public:
        enum class Order {
            COLUMN_MAJOR,
            ROW_MAJOR
        };

        struct Strides {
            int row;
            int col;
            int ch;
        };


    private:
        static Strides compute_strides(int m, int n, Order order);

        // following the convention of using _ as a suffix for internal variables
        std::span<T, Extent> data;
        Shape shape_;
        Order order_;
        Strides strides_;
        int size_;
    public:
        // declaring getters here, because the verbosity of moving getters to the impl file seems excessive
        const Shape& shape() const {return this->shape_;}
        const Strides& strides() const {return this->strides_;}
        Order order() const {return this->order_;}
        int size() const {return this->size_;} // TODO: should size be m*n or m*n*channels

        Multichannel_Matrix(std::span<T, Extent> data, int m, int n, Order order = Order::ROW_MAJOR);

        // need a constructor that will match the default constructor so a variable can be declared before assignment
        // Such functionality is very useful for a view object
        Multichannel_Matrix() : Multichannel_Matrix({}, 0, 0) {}

        // TODO: is there a better name for this operator?
        T& index(int i, int j, int k);
        const T& index(int i, int j, int k) const;
        T& index(int i); // direct index on underlying contiguous memory, perhaps use [] instead here?
        const T& index(int i) const;

        Multichannel_Matrix<T, 1> channel(int k);
};

// template<typename T>
// class Multichannel_Matrix<T, 1> {
//     public:
//         T& index(int i, int j) {
//             return this->data[i * this->strides.row + j * this->strides.col]; 
//         }

//         const T& index(int i, int j) const {
//             return this->data[i * this->strides.row + j * this->strides.col]; 
//         }
// };

template<typename T, std::size_t Extent = std::dynamic_extent>
using Matrix = Multichannel_Matrix<T, 1, Extent>;

template<typename T, std::size_t Extent = std::dynamic_extent>

using Image_View = Multichannel_Matrix<T, 3, Extent>;

template<typename T, typename U>
class Abstract_Image_Transformer : public Abstract_Transformer<T, T, Image_View> {
    
    public:
        const Shape input_shape;
        const Shape output_shape;

        Abstract_Image_Transformer(const Shape shape)
            : input_shape(shape), output_shape(shape), 
            Abstract_Transformer<T, T, Image_View>(
                shape.m * shape.n * 3) {}
        
        Abstract_Image_Transformer(const Shape input_shape, const Shape output_shape)
            : input_shape(input_shape), output_shape(output_shape),
            Abstract_Transformer<T, T, Image_View>(
                input_shape.m * input_shape.n * 3, output_shape.m * output_shape.n * 3) {}

};





template<typename T>
class YCbCr_Transformer : public Abstract_Image_Transformer<T, T> {
    public:
        static const int Channels = 3;
    private:
        std::array<double, 9> forward_transform_data;
        std::array<double, 9> inverse_transform_data;
    public:
        const float k_r;
        const float k_g;
        const float k_b;
        Multichannel_Matrix<T, 1> transform_matrix;


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
        int transform(Image_View<const T> in, Image_View<T> out) override;
        int inverse(Image_View<const T> in, Image_View<T> out) override {return -1;}

        std::array<T, 9> get_forward_transform(float k_r, float k_g, float k_b);
};

#include "multichannel_matrix_impl.hpp"
#include "ycbcr_transformer_impl.hpp"

#endif