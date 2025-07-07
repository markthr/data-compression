#ifndef __MATRIX_OPERATIONS_IMPL_H__
#define __MATRIX_OPERATIONS_IMPL_H__

#include <vector>
#include <utility>
#include "2d_transforms.hpp"

// TODO: initialize zero or split loop to initialize before accumulate product?
// TODO: this should return a matrix. Returning a pair is clunky (see channel transformer) so making a version of matrix which is an owning type would be good.
namespace mat {
    template<typename T, size_t Extent1, size_t Extent2>
    std::vector<T> multiply(Matrix<T, Extent1> m1, Matrix<T, Extent2> m2) {
        assert(m1.shape().n && m1.shape().n == m2.shape().m); // no reason to have an exception that gets handled, bad matrix multiplication is bad code and not an exceptional case

        std::vector<T> product(m1.shape().m * m2.shape().n);

        for(int m1_i = 0; m1_i < m1.shape().m; m1_i++) {
            for(int m2_j = 0; m2_j < m2.shape().n; m2_j++) {
                // currently assuming row major for output
                int index = m1_i * m2.shape().n + m2_j;
                // m1 * m2 has shape (m1.shape().m, m2.shape.n())
                product[index] = m1.index(m1_i, 0, 0) * m2.index(0, m2_j, 0);
                for(int k = 1; k < m1.shape().n; k++) {
                    product[index] += m1.index(m1_i, k, 0) * m2.index(k, m2_j, 0);
                }
            }
        }

        return product;
    }


    template<typename T, int Channels, size_t Extent>
    int transform_channels(Multichannel_Matrix<T, Channels, Extent> input,
            Multichannel_Matrix<T, 1, Channels * Channels> transform,
            Multichannel_Matrix<T, Channels, Extent> output) {
        
        // ensure transform is a square matrix and output is of sufficient size
        if(transform.shape().m != Channels || transform.shape().n != Channels || input.size() != output.size()){
            return -1;
        }
        output.reshape(input.shape());
        
        std::array<T, Channels> pixel_data;
        Matrix<T, Channels> pixel_mat(std::span<T, Channels>(pixel_data), {Channels, 1});

        for(int i = 0; i < input.shape().m; i++) {
            for(int j = 0; j < input.shape().n; j++) {
                for(int ch = 0; ch < Channels; ch++) {
                    pixel_data[ch] = input.index(i, j, ch); // fetch data for pixel
                }

                std::vector<T> product = mat::multiply(transform, pixel_mat);

                // TODO: his can be optimized to combine the loops, is this worth optimizing?
                for(int ch = 0; ch < Channels; ch++) {
                    output.index(i, j, ch) = product[ch]; // save result from transformed pixel
                }
            }   
        }

        return 0;
    }
 
    /**
     * Use a square matrix to transform a matrix's channels
     * 
     * TODO: this operation naturally wants to return a Multichannel_Matrix, but that is a non-owning type
     * The use of a pair is not very elegant, perhaps allowing both owning and non-owning matrices would be good
     */
    template<typename T, int Channels, size_t Extent=std::dynamic_extent>
    std::pair<std::vector<T>, Multichannel_Matrix<T, Channels, Extent>> transform_channels (Multichannel_Matrix<T, Channels, Extent> m1, Matrix<T, Channels * Channels> transform) {
        std::vector<T> product(m1.size(), 0);
        Multichannel_Matrix<T, Channels, Extent> transformed_matrix(product, m1.shape().m, m1.shape().n, m1.order());

        mat::transform_channels(m1, transform, transformed_matrix);
        
        return std::pair(product, transformed_matrix);
    }
}

#endif