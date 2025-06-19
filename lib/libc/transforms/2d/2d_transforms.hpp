#ifndef __2D_TRANSFORMS_H__
#define __2D_TRANSFORMS_H__

#include <span>
#include "../transforms.hpp"

struct Shape {
    int m; // m=height=# of rows
    int n; // n=width=# of cols
};

template<typename T, int Channels>
class Multichannel_Matrix {
    public:
        enum class Order {
            COLUMN_MAJOR,
            ROW_MAJOR
        };


    private:
        const int row_stride;
        const int col_stride;
        const int ch_stride;
        std::span<T> data;
    public:
        const Shape shape;
        const Order order;

        Multichannel_Matrix(std::span<T> data, int m, int n, Order order = Order::ROW_MAJOR);

        T& index(int i, int j, int k);
        Multichannel_Matrix<T, 1> channel(int k);
};

template<typename T>
using Matrix = Multichannel_Matrix<T, 1>;

template<typename T, typename U, int Channels>
class Abstract_MCM_Transformer : public Abstract_Transformer<
        Multichannel_Matrix<T, Channels>,
        Multichannel_Matrix<T, Channels>> {
    
    public:
        const Shape input_shape;
        const Shape output_shape;

        Abstract_MCM_Transformer(const Shape shape)
            : input_shape(shape), output_shape(shape), 
            Abstract_Transformer<Multichannel_Matrix<T, Channels>, Multichannel_Matrix<T, Channels>>(
                shape.m * shape.n * Channels) {}
        
        Abstract_MCM_Transformer(const Shape input_shape, const Shape output_shape)
            : input_shape(input_shape), output_shape(output_shape),
            Abstract_Transformer<Multichannel_Matrix<T, Channels>,Multichannel_Matrix<T, Channels>>(
                input_shape.m * input_shape.n, output_shape.m * output_shape.n) {}

};



template<typename T>
using Image_Vew = Multichannel_Matrix<T, 3>;

template<typename T>
class YCbCr_Transformer : public Abstract_MCM_Transformer<T, T, 3> {
    public:
        const Shape shape;
        YCbCr_Transformer(const Shape shape) : shape(shape) {}
        // indicate override of pure virtual signature
        int transform(std::span<const T> in, std::span<T> out) override {return -1;}
        int inverse(std::span<const T> in, std::span<T> out) override {return -1;}
};

#include "multichannel_matrix_impl.hpp"

#endif