#ifndef __MATRIX_OPERATIONS_IMPL_H__
#define __MATRIX_OPERATIONS_IMPL_H__

#include <vector>
#include "2d_transforms.hpp"

namespace mat {
    template<typename T, size_t Extent=std::dynamic_extent>
    std::vector<T> multiply(Matrix<T, Extent> m1, Matrix<T, Extent> m2) {
        assert(m1.shape().n && m1.shape().n == m2.shape().m); // no reason to have an exception that gets handled, bad matrix multiplication is bad code and not an exceptional case

        std::vector<T> product(m1.shape().m * m2.shape().n);

        for(int m1_i = 0; m1_i < m1.shape().m; m1_i++) {
            for(int m2_j = 0; m2_j < m2.shape().n; m2_j++) {
                int index = m1_i * m1.shape().n + m2_j;
                product[index] = m1.index(m1_i, 0, 0) * m2.index(0, m2_j, 0);
                for(int k = 1; k < m1.shape().m; k++) {
                    product[index] += m1.index(m1_i, k, 0) * m2.index(k, m2_j, 0);
                }
            }
        }

        return product;
    }
}

#endif