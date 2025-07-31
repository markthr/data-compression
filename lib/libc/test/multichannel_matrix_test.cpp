#include <gtest/gtest.h>
#include <initializer_list>



// TODO: is there a better way of handling this import?
#include "../transforms/2d/2d_transforms.hpp"

TEST(MultichannelMatrixTest, reshape) {
    std::initializer_list<int> data1 {1, 2, 3, 4, 5, 6};
    const Shape shape1 = {2, 3};
    const Shape shape2 = {3, 2};
    Matrix<int> mat1(data1, shape1);

    Matrix<int> mat2 = mat1.reshape(shape2);
    Matrix<int> mat2_expected(data1, shape2);

    for(int i = 0; i < data1.size(); i++) {
        EXPECT_EQ(mat1[i], mat2[i]) << " at index: " << i << ", reshape() should not change underlying data";
    }

    for(int i = 0; i < shape2.m; i++) {
        for(int j = 0; j < shape2.n; j++) {
            EXPECT_EQ(mat2.index(i, j), mat2_expected.index(i, j )) << " at index: (" << i << ", " << j
                    << "), incorrect reshape(...) output";
        }
    }
}

TEST(MultichannelMatrixTest, transpose) {
    std::initializer_list<float> data1 {7, 5, 3, 2, 4, 1};
    const Shape shape1 = {3, 2};
    Matrix<float> mat1 (data1, shape1);

    std::initializer_list<float> data1_T {7, 3, 4, 5, 2, 1};
    const Shape shape1_T = {2, 3};
    Matrix<float> mat1_T_expected (data1_T, shape1_T);

    Matrix<float> mat1_T = mat1.transpose();

    for(int i = 0; i < shape1_T.m; i++) {
        for(int j = 0; j < shape1_T.n; j++) {
            EXPECT_EQ(mat1_T.index(i, j), mat1_T_expected.index(i, j )) << " at index: (" << i << ", " << j
                    << "), incorrect transpose() output";
        }
    }

}

TEST(MultichannelMatrixTest, jagged) {
    const int CHANNELS_1 = 3;
    std::initializer_list<int> data_1 = {1, 2, 3, 1, 0, 0, 1, -5, -4, -3, -2, -1};
    Jagged_Multichannel_Matrix<int, CHANNELS_1> jmat_1 (data_1, {{3, 1}, {2, 2}, {1, 5}});

    EXPECT_EQ(12, jmat_1.size()) << " expected size of jagged mat to be equal to sum of submatrix shapes";

    std::array<Matrix<int>, CHANNELS_1> exp1_arr {
        Matrix<int> ({1, 2, 3}, {3,1}),
        mat::eye<int, 1>(2),
        Matrix<int> ({-5, -4, -3, -2, -1}, {3,1})
    };
    
    for(int k = 0; k < CHANNELS_1; k++) {
        for(int i = 0; i < jmat_1.channel_view(k).shape().m; i++) {
            for(int j = 0; j < jmat_1.channel_view(k).shape().n; j++) {
                EXPECT_EQ(exp1_arr[k].index(i, j), jmat_1.channel_view(k).index(i, j)) << " at index: (" << i << ", " << j << ", " << k << ")";
                EXPECT_EQ(exp1_arr[k].index(i, j), jmat_1.index(i, j, k)) << " at index: (" << i << ", " << j << ", " << k << ")";
            }
        }
    }

    const int CHANNELS_2 = 4;
    std::initializer_list<short> data_2 = {16, 5, 2, 4, 2, 7, 10, -1, 15, 1, 11, 16, 5, -13, -5, 7, 15, 17, -3, 4, 11,
            14, 2, 2, 6, 19, -10, 16, 0, -12, -18, -11, -5, 4, 17, -9};
    
    Multichannel_Matrix<short, CHANNELS_2> mat_2 (data_2, {3, 3}, Order::ROW_COL_CH);
    auto jmat_2 = Jagged_Multichannel_Matrix<short, CHANNELS_2>::as_jagged(mat_2);

    EXPECT_EQ(jmat_2.size(), mat_2.size());

    for(int k = 0; k < CHANNELS_2; k++) {
        EXPECT_EQ(jmat_2.channel_view(k).shape().m, mat_2.shape().m);
        EXPECT_EQ(jmat_2.channel_view(k).shape().n, mat_2.shape().n);
        for(int i = 0; i < mat_2.shape().m; i++) {
            for(int j = 0; j < mat_2.shape().n; j++) {
                EXPECT_EQ(mat_2.index(i, j, k), jmat_2.index(i, j, k)) << " at index: (" << i << ", " << j << ", " << k << ")";
            }
        }
    }   

    


    
}