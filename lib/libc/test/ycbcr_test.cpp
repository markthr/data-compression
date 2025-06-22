#include <gtest/gtest.h>

// TODO: is there a better way of handling this import?
#include "../transforms/2d/2d_transforms.hpp"

/**
 * Simple assertions to verify forward and inverse transform return wtihout error
 */
TEST(YCbCrTest, TransformMatrices) {
    Shape shape = {32, 32};
    YCbCr_Transformer<double> ytr(shape);

    EXPECT_EQ(ytr.transform_matrix.index(0, 0, 0), ytr.k_r) << "Coeffs not equal";
}