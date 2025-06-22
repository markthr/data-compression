#ifndef __MULTICHANNEL_MATRIX_IMPL_H__
#define __MULTICHANNEL_MATRIX_IMPL_H__

#include "2d_transforms.hpp"

#include <span>



template<typename T, int Channels, std::size_t Extent>
Multichannel_Matrix<T, Channels, Extent>::Multichannel_Matrix(std::span<T, Extent> data, int m, int n, Order order) 
        : data(data), shape_({m, n}), order_(order), size_(m*n*Channels), strides_(Multichannel_Matrix<T, Channels, Extent>::compute_strides(m, n, order)){}

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

// TODO: is there a way to make this type signature more concise? could alias the namespace if that isn't a terrible idea
//      could also combine the function definitions with the class definition
template<typename T, int Channels, std::size_t Extent>
Multichannel_Matrix<T, Channels, Extent>::Strides Multichannel_Matrix<T, Channels, Extent>::compute_strides(int m, int n, Order order) {
    Strides strides;

    if(order == Order::ROW_MAJOR) {
        strides.row = 1;
        strides.col = n;
    }
    else if(order == Order::COLUMN_MAJOR) { // unnecesary check, but it is cheap and adds clarity
        strides.row = m;
        strides.col = 1;
    }
    if(Channels > 1) {
        strides.ch = Channels * m * n;
    }
    else {
        strides.ch = 0; // TODO: is there a better way to handle single channel matrices?
    }

    return strides;
}
#endif