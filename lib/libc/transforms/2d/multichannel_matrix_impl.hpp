#ifndef __MULTICHANNEL_MATRIX_IMPL_H__
#define __MULTICHANNEL_MATRIX_IMPL_H__

#include "2d_transforms.hpp"

#include <span>



template<typename T, int Channels>
Multichannel_Matrix<T, Channels>::Multichannel_Matrix(std::span<T> data, int m, int n, Order order) 
        : shape({m, n}), order(order), row_stride(order == Order::ROW_MAJOR ? 1 : m),
        col_stride(order == Order::COLUMN_MAJOR ? 1 : n), ch_stride(m*n){}

template<typename T, int Channels>
T& Multichannel_Matrix<T, Channels>::index(int i, int j, int k) {
    return this->data[i * this->row_stride + j * this->col_stride + k * this->ch_stride];
}

template<typename T, int Channels>
Matrix<T> Multichannel_Matrix<T, Channels>::channel(int k) {
    return Multichannel_Matrix<T, 1>(this->data.subspan(this->ch_stride * k, this->ch_stride));
}
#endif