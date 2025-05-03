#include <gtest/gtest.h>

// TODO: is there a better way of handling this import?
#include "../transforms/transforms.hpp"

const double PI = std::numbers::pi;
const double ERR = 1E-8f;

/**
 * Simple assertions to verify forward and inverse transform return wtihout error
 */
TEST(DiscreteFourierTest, BasicAssertions) {
    // test template with float
    const int size = 8;
    std::vector<float> input_float(size);
    std::vector<std::complex<float>> output_float;

    FFT<float, size> fft_float;
    EXPECT_EQ(fft_float.transform(input_float, output_float), 0) << "Forward transform failed for float";
    EXPECT_EQ(fft_float.inverse(output_float, input_float), 0) << "Inverse transform failed for float";

    // test template with double
    std::vector<double> input_double(size);
    std::vector<std::complex<double>> output_double;

    FFT<double, size> fft_double;
    EXPECT_EQ(fft_double.transform(input_double, output_double), 0) << "Forward transform failed for double";
    EXPECT_EQ(fft_double.inverse(output_double, input_double), 0) << "Inverse transform failed for double";
}

/**
 * 8-point FFT tests of forward transform
 */
TEST(DiscreteFourierTest, BasicTransforms) {
    const int size = 8;
    FFT<float, size> fft;
    std::vector<std::complex<float>> output;

    // compute FFT of constant signal
    const std::vector<float> input_1(size, 1);
    const std::vector<float> exp_mag_1 = {size, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<float> exp_phase_1(size, 0);

    EXPECT_FALSE(fft.transform(input_1, output));

    for(int k = 0; k < size; k++) {
        EXPECT_FLOAT_EQ(std::abs(output[k]), exp_mag_1[k]) << "at index: " << k;
        EXPECT_FLOAT_EQ(std::arg(output[k]), exp_phase_1[k]) << "at index: " << k;
    }

    // compute FFT of an impulse
    std::vector<float> input_2(size, 0);
    input_2[2] = 1;
    const std::vector<float> exp_mag_2(size, 1);
    const std::vector<float> exp_phase_2 = {0, -PI/2, PI, PI/2, 0, -PI/2, PI, PI/2};

    EXPECT_FALSE(fft.transform(input_2, output));

    for(int k = 0; k < size; k++) {
        EXPECT_FLOAT_EQ(std::abs(output[k]), exp_mag_2[k]) << "at index: " << k;
        EXPECT_FLOAT_EQ(std::arg(output[k]), exp_phase_2[k]) << "at index: " << k;
    }

    // compute FFT of a more complex signal
    const std::vector<float> input_3 = {0, 6, -1, 3, 3, 0, -5, 2};
    const std::vector<float> exp_mag_3 = {8, 8.9657558, 9.0553851, 6.6041823, 14, 6.6041823, 9.0553851, 8.9657558};
    const std::vector<float> exp_phase_3 = {0*PI, -0.4809757*PI, -0.035223287*PI, -0.95406458*PI, 1*PI, 0.95406458*PI, 0.035223287*PI, 0.4809757*PI};

    EXPECT_FALSE(fft.transform(input_3, output));

    for(int k = 0; k < size; k++) {
        EXPECT_FLOAT_EQ(std::abs(output[k]), exp_mag_3[k]) << "at index: " << k;
        EXPECT_FLOAT_EQ(std::arg(output[k]), exp_phase_3[k]) << "at index: " << k;
    }
}

/**
 * FFT tests of forward transform for larger input
 */
TEST(DiscreteFourierTest, LargeTransforms) {
    const int size_1 = 16;
    FFT<double, size_1> fft_16;
    std::vector<std::complex<double>> output;

    // compute 16-point FFT
    const std::vector<double> input_1 = {0.19325677, 0.50802583, 0.43814404, 0.32752371, 0.95930142, 0.064225197, 0.3702718, 0.7806024, 0.92368416, 0.15440416,
        0.36631957, 0.6990894, 0.65803902, 0.76735618, 0.5950245, 0.49130181};
    const std::vector<double> exp_mag_1 = {8.29657, 0.64687173, 0.63673424, 1.3593968, 1.2559982, 1.5603724, 0.56979207, 1.1762005, 0.71151258, 1.1762005,
        0.56979207, 1.5603724, 1.2559982, 1.3593968, 0.63673424, 0.64687173};
    const std::vector<double> exp_phase_1 = {0*PI, 0.67293605*PI, 0.74740989*PI, -0.85557327*PI, 0.22128576*PI, -0.63866057*PI, 0.92545546*PI,
        0.7052744*PI, 0*PI, -0.7052744*PI, -0.92545546*PI, 0.63866057*PI, -0.22128576*PI, 0.85557327*PI, -0.74740989*PI, -0.67293605*PI};
    
    EXPECT_FALSE(fft_16.transform(input_1, output));

    for(int k = 0; k < size_1; k++) {
        EXPECT_DOUBLE_EQ(std::abs(output[k]), exp_mag_1[k]) << "at index: " << k;
        EXPECT_DOUBLE_EQ(std::arg(output[k]), exp_phase_1[k]) << "at index: " << k;
    }
}

/**
 * FFT test for inverse of transform being equivalent to identity
 */
TEST(DiscreteFourierTest, IdentityTransforms) {
    std::vector<std::complex<double>> transformed;
    std::vector<double> output;

    // test 8-point FFT
    const int size_1 = 8;
    FFT<double, size_1> fft_8;
    const std::vector<double> input_1 = {-0.67734518, 0.041097089, -0.83970564, -0.76808002, 0.28387797, -0.047668902, 0.70266354, 0.78609125};

    EXPECT_FALSE(fft_8.transform(input_1, transformed));
    EXPECT_FALSE(fft_8.inverse(transformed, output));

    for(int k = 0; k < size_1; k++) {
        EXPECT_NEAR(input_1[k], output[k], ERR) << "at index: " << k;
    }

    

    // test 16-point FFT
    const int size_2 = 16;
    FFT<double, size_2> fft_16;
    const std::vector<double> input_2 = {-0.80143361, -4.1420611, 4.7281667, 3.655163, 2.2732652, 0.51822452, -1.5725374, -0.41392366, 3.1228435, -4.8941613, 1.8463468, -0.29013594, -2.5307239, 4.2880149, 1.8248971, 0.0077635051};

    EXPECT_FALSE(fft_16.transform(input_2, transformed));
    EXPECT_FALSE(fft_16.inverse(transformed, output));

    for(int k = 0; k < size_2; k++) {
        EXPECT_NEAR(input_2[k], output[k], ERR) << "at index: " << k;
    }
}