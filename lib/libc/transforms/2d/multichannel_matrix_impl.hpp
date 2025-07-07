#ifndef __MULTICHANNEL_MATRIX_IMPL_H__
#define __MULTICHANNEL_MATRIX_IMPL_H__

#include "2d_transforms.hpp"

#include <span>


template<typename T, int Channels, std::size_t Extent>
Multichannel_Matrix<T, Channels, Extent>::Multichannel_Matrix(std::span<T, Extent> data, Shape shape, Strides strides, Order order)
        : data(data), shape_(shape), order_(order), size_(shape.m*shape.n*Channels), strides_(strides) {}

template<typename T, int Channels, std::size_t Extent>
Multichannel_Matrix<T, Channels, Extent>::Multichannel_Matrix(std::span<T, Extent> data, Shape shape, Order order) 
        : Multichannel_Matrix(data, shape, compute_strides(shape, order), order) {}

template<typename T, int Channels, std::size_t Extent>
T& Multichannel_Matrix<T, Channels, Extent>::index(int i, int j, int k) {
    return this->data[i * this->strides_.row + j * this->strides_.col + k * this->strides_.ch];
}

template<typename T, int Channels, std::size_t Extent>
const T& Multichannel_Matrix<T, Channels, Extent>::index(int i, int j, int k) const {
    return this->data[i * this->strides_.row + j * this->strides_.col + k * this->strides_.ch];
}


template<typename T, int Channels, std::size_t Extent>
T& Multichannel_Matrix<T, Channels, Extent>::index(int i) {
    return this->data[i];
}

template<typename T, int Channels, std::size_t Extent>
const T& Multichannel_Matrix<T, Channels, Extent>::index(int i) const {
    return this->data[i];
}

template<typename T, int Channels, std::size_t Extent>
Matrix<T> Multichannel_Matrix<T, Channels, Extent>::channel(int k) {
    return Multichannel_Matrix<T, 1>(this->data.subspan(this->strides_.ch * k, this->strides_.ch));
}

template<typename T, int Channels, std::size_t Extent>
void Multichannel_Matrix<T, Channels, Extent>::reshape(Shape shape) {
    assert(this->shape().m * this->shape().n == shape.m * shape.n); // ensure size does not change
    this->shape_ = shape;
}


// TODO: is there a way to make this type signature more concise? could alias the namespace if that isn't a terrible idea
//      could also combine the function definitions with the class definition
template<typename T, int Channels, std::size_t Extent>
Strides Multichannel_Matrix<T, Channels, Extent>::compute_strides(Shape shape, Order order) {
    // ensure dimensions are not repeated in the ordering
    assert(order.first != order.second && order.first != order.third && order.second != order.third); 
    Strides strides;
    
    int multiplier = 1;

    const int n_dim = 3;
    Dimension dims[n_dim] = {order.first, order.second, order.third};
    for(int d = 0; d < n_dim; d++) {
        if(dims[d] == Dimension::ROW) {
            strides.col = multiplier;
            multiplier *= shape.n;
        } else if(dims[d] == Dimension::COLUMN) {
            strides.row = multiplier;
            multiplier *= shape.m;
        }
        else { // must be Dimension::Channel
            strides.ch = multiplier;
            multiplier = Channels;
        }
    }

    return strides;
}

#endif