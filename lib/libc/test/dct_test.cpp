#include <gtest/gtest.h>

// TODO: is there a better way of handling this import?
#include "../transforms/transforms.hpp"

const double PI = std::numbers::pi;

/**
 * Simple assertions to verify forward and inverse transform return wtihout error
 */
TEST(DiscreteCosineTest, BasicAssertions) {
    const double ERR = 1E-6f;
    // test template with float
    const int size = 8;
    std::vector<float> input_float(size);
    std::vector<float> output_float;

    DCT_2<float, size> dct2_float;
    EXPECT_EQ(dct2_float.transform(input_float, output_float), 0) << "Forward transform failed for float";
    EXPECT_EQ(dct2_float.inverse(output_float, input_float), 0) << "Inverse transform failed for float";

    // test template with double
    std::vector<double> input_double(size);
    std::vector<double> output_double;

    DCT_2<double, size> dct2_double;
    EXPECT_EQ(dct2_double.transform(input_double, output_double), 0) << "Forward transform failed for double";
    EXPECT_EQ(dct2_double.inverse(output_double, input_double), 0) << "Inverse transform failed for double";
}

/**
 * 8-point DCT tests of forward transform
 * 
 * Test values from running mathematica code on wolfram alpha which only gives 5 decimals
 * for the test input (6 digits) so error of 1E-5
 * 
 * Note that mathematica uses a convention equivalent to multiplying every element of the
 * output by beta_0
 */
TEST(DiscreteCosineTest, BasicTransforms) {
    const double ERR = 1E-5;
    const int size = 8;

    DCT_2<float, size> dct;
    std::vector<float> output;

    // compute FFT of constant signal
    const std::vector<float> input_1(size, 1);
    const std::vector<float> exp_1 = {std::sqrt(size), 0, 0, 0, 0, 0, 0, 0};

    EXPECT_FALSE(dct.transform(input_1, output));

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(output[k], exp_1[k], ERR) << "at index: " << k;
    }

    // compute FFT of an impulse
    std::vector<float> input_2(size, 0);
    input_2[2] = 1;
    const std::vector<float> exp_2 = {0.353553, 0.277785, -0.191342, -0.490393, -0.353553, 0.0975451, 0.46194, 0.415735};

    EXPECT_FALSE(dct.transform(input_2, output));

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(output[k], exp_2[k], ERR) << "at index: " << k;
    }

    // compute FFT of a more complex signal
    const std::vector<float> input_3 = {0, 6, -1, 3, 3, 0, -5, 2};
    const std::vector<float> exp_3 = {2.82843, 3.31451, -1.46507, -1.41407, 2.82843, -6.04743, -1.68925, -3.66646};

    EXPECT_FALSE(dct.transform(input_3, output));

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(output[k], exp_3[k], ERR) << "at index: " << k;
    }
}

/**
 * DCT test for inverse of transform being equivalent to identity
 */
TEST(DiscreteCosineTest, IdentityTransforms) {
    const double ERR = 1E-6f;
    std::vector<double> transformed;
    std::vector<double> output;

    // test 8-point DCT_2
    const int size_1 = 8;
    DCT_2<double, size_1> dct2_8;
    const std::vector<double> input_1 = {-0.67734518, 0.041097089, -0.83970564, -0.76808002, 0.28387797, -0.047668902, 0.70266354, 0.78609125};

    EXPECT_FALSE(dct2_8.transform(input_1, transformed));
    EXPECT_FALSE(dct2_8.inverse(transformed, output));

    for(int k = 0; k < size_1; k++) {
        EXPECT_NEAR(input_1[k], output[k], ERR) << "at index: " << k;
    }

    

    // test 16-point DCT_2
    const int size_2 = 16;
    DCT_2<double, size_2> dct2_16;
    const std::vector<double> input_2 = {-0.80143361, -4.1420611, 4.7281667, 3.655163, 2.2732652, 0.51822452, -1.5725374, -0.41392366, 3.1228435, -4.8941613, 1.8463468, -0.29013594, -2.5307239, 4.2880149, 1.8248971, 0.0077635051};

    EXPECT_FALSE(dct2_16.transform(input_2, transformed));
    EXPECT_FALSE(dct2_16.inverse(transformed, output));

    for(int k = 0; k < size_2; k++) {
        EXPECT_NEAR(input_2[k], output[k], ERR) << "at index: " << k;
    }
}