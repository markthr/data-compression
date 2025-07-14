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
        EXPECT_EQ(mat1.index(i), mat2.index(i)) << " at index: " << i << ", reshape() should not change underlying data";
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