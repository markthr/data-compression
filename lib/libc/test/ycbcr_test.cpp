#include <gtest/gtest.h>
#include <array>
#include <fstream>
#include <string>
#include <string_view>
#include <charconv>
#include <sstream>


// TODO: is there a better way of handling this import?
#include "../transforms/2d/2d_transforms.hpp"
#include "image_file_io.hpp"

/**
 * Simple assertions to verify that transform matrices are set correctly
 */
TEST(YCbCrTest, TransformMatrices) {
    const float ERR = 1e-4; // only have 3 decimals from Wolfram outputs
    const int size = 9;
    Shape shape = {32, 32};
    YCbCr_Transformer<double> ytr(shape, 0.114, 0.299);

    std::array<float, size> exp_forward  = {0.299, 0.587, 0.114, -0.168736, -0.331264, 0.5, 0.5, -0.418688, -0.081312};

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(exp_forward[k], ytr.transform_matrix.index(k), ERR) << "Coeffs not equal at index: " << k;
    }

    auto identity = mat::multiply(ytr.inverse_matrix, ytr.transform_matrix);
    auto exp_identity = mat::eye<float>(3);

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(exp_identity.index(k), identity.index(k), ERR) << "Coeffs not equal at index: " << k;
    }

    const int CHANNELS = 3;
    std::array<double, CHANNELS> exp_product = {191.979, -16.9182, 22.1262};
    Matrix<double> pixel_mat(std::initializer_list<double>{223, 182, 162}, {CHANNELS, 1});
    Matrix<double> product = mat::multiply(ytr.transform_matrix, pixel_mat);

    for(int k = 0; k < CHANNELS; k++) {
        EXPECT_NEAR(exp_product[k], product.index(k), ERR) << "Product not equal at index: " << k;
    }
}

TEST(YCbCrTest, IdentityTransforms) {
    const float ERR = 1e-4;
    const Shape SHAPE{327, 223};
    const int CHANNELS = 3;
    const int SIZE = SHAPE.m * SHAPE.n * CHANNELS;

    auto image = img::read_img_csv<float>("data/orioles_mascot_cropped.csv", SHAPE, Order::CH_ROW_COL);

    if(image.size() == 0) {
        FAIL() << "Unable to open test data";
    }
    else {
        EXPECT_EQ(image.size(), SIZE) << "Failed to parse test data";
        
        Image_Matrix<float> transformed(SHAPE, Order::CH_ROW_COL);
        Image_Matrix<float> output(SHAPE, Order::CH_ROW_COL);

        YCbCr_Transformer<float> ycbcr(SHAPE);
        EXPECT_FALSE(ycbcr.transform(image, transformed)) << "Forward transform failed";

        for(int i = 0; i < image.shape().m && i < image.shape().n; i++) {
            for(int k = 0; k < CHANNELS; k++) {
                if(image.index(i, i, k)) {
                    EXPECT_NE(image.index(i, i, k), transformed.index(i, i, k));
                }
            }
        }

        EXPECT_FALSE(ycbcr.inverse(transformed, output))  << "Inverse transform failed";

        for(int n = 0; n < image.size(); n++) {
            if(std::abs(image.index(n) - output.index(n)) > ERR) {
                EXPECT_NEAR(image.index(n), output.index(n), ERR) << "Transformed value not equal at index: " << n;
                break;
            }
        }
    }
}