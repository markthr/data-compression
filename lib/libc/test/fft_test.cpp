#include <gtest/gtest.h>

// TODO: is there a better way of handling this import?
#include "../transforms/transforms.hpp"

const float PI = std::numbers::pi;

// Demonstrate some basic assertions.
TEST(DiscreteFourierTest, BasicAssertions) {
    const int size = 8;
    std::vector<float> input(size);
    std::vector<std::complex<float>> output;

    FFT<float, size> fft;
    EXPECT_EQ(fft.transform(input, output), 0) << "Forward transform failed";
    EXPECT_EQ(fft.inverse(output, input), 0) << "Inverse transform failed";
}

/**
 * 8-point FFT tests of forward transform
 */
TEST(DiscreteFourierTest, BasicTransforms) {
    const int size = 8;
    FFT<float, size> fft;
    std::vector<std::complex<float>> output;

    // compute FFT of constant signal
    std::vector<float> input_1(size, 1);
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
    std::vector<float> input_3 = {0, 6, -1, 3, 3, 0, -5, 2};
    const std::vector<float> exp_mag_3 = {8, 8.9657558, 9.0553851, 6.6041823, 14, 6.6041823, 9.0553851, 8.9657558};
    const std::vector<float> exp_phase_3 = {0*PI, -0.4809757*PI, -0.035223287*PI, -0.95406458*PI, 1*PI, 0.95406458*PI, 0.035223287*PI, 0.4809757*PI};

    EXPECT_FALSE(fft.transform(input_3, output));

    for(int k = 0; k < size; k++) {
        EXPECT_FLOAT_EQ(std::abs(output[k]), exp_mag_3[k]) << "at index: " << k;
        EXPECT_FLOAT_EQ(std::arg(output[k]), exp_phase_3[k]) << "at index: " << k;
    }
}