#include <gtest/gtest.h>
#include <array>

// TODO: is there a better way of handling this import?
#include "../transforms/2d/2d_transforms.hpp"

/**
 * Simple assertions to verify that transform matrices are set correctly
 */
TEST(YCbCrTest, TransformMatrices) {
    const float ERR = 1e-6;
    const int size = 9;
    Shape shape = {32, 32};
    YCbCr_Transformer<double> ytr(shape, 0.114, 0.299);

    std::array<float, size> exp_forward  = {0.299, 0.587, 0.114, -0.168736, -0.331264, 0.5, 0.5, -0.418688, -0.081312};

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(exp_forward[k], ytr.transform_matrix.index(k), ERR) << "Coeffs not equal at index: " << k;
    }

    auto identity = mat::multiply(ytr.inverse_matrix, ytr.transform_matrix);
    std::array<float, size> exp_identity  = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(exp_forward[k], ytr.transform_matrix.index(k), ERR) << "Coeffs not equal at index: " << k;
    }
}