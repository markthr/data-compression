#include <gtest/gtest.h>

// TODO: is there a better way of handling this import?
#include "../transforms/transforms.hpp"

const double PI = std::numbers::pi;
const double ERR = 1E-6f;

/**
 * Simple assertions to verify forward and inverse transform return wtihout error
 */
TEST(DiscreteFourierTest, BasicAssertions) {
    // test template with float
    const int size = 8;
    std::vector<float> input_float(size);
    std::vector<std::complex<float>> output_float(size);

    FFT<float> fft_float(size);
    EXPECT_EQ(fft_float.transform(input_float, output_float), 0) << "Forward transform failed for float";
    EXPECT_EQ(fft_float.inverse(output_float, input_float), 0) << "Inverse transform failed for float";

    // test template with double
    std::vector<double> input_double(size);
    std::vector<std::complex<double>> output_double(size);

    FFT<double> fft_double(size);
    EXPECT_EQ(fft_double.transform(input_double, output_double), 0) << "Forward transform failed for double";
    EXPECT_EQ(fft_double.inverse(output_double, input_double), 0) << "Inverse transform failed for double";

    std::vector<std::complex<double>> output_double_2 = fft_double.transform(input_double);
    EXPECT_EQ(output_double_2.size(), size) << "Vector return for forward transform failed";
    std::vector<double> input_double_2 = fft_double.inverse(output_double_2);
    EXPECT_EQ(input_double_2.size(), size) << "Vector return for inverse transform failed";
}

/**
 * 8-point FFT tests of forward transform
 */
TEST(DiscreteFourierTest, BasicTransforms) {
    const int size = 8;
    FFT<float> fft(size);

    // compute FFT of constant signal
    const std::vector<float> input_1(size, 1);
    const std::vector<float> exp_mag_1 = {size, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<float> exp_phase_1(size, 0);

    std::vector<std::complex<float>> output = fft.transform(input_1);

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(std::abs(output[k]), exp_mag_1[k], ERR) << "at index: " << k;
        EXPECT_NEAR(std::arg(output[k]), exp_phase_1[k], ERR) << "at index: " << k;
    }

    // compute FFT of an impulse
    std::vector<float> input_2(size, 0);
    input_2[2] = 1;
    const std::vector<float> exp_mag_2(size, 1);
    const std::vector<float> exp_phase_2 = {0, -PI/2, 1*PI, PI/2, 0, -PI/2, 1*PI, PI/2};

    EXPECT_FALSE(fft.transform(input_2, output));

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(std::abs(output[k]), exp_mag_2[k], ERR) << "at index: " << k;
        EXPECT_NEAR(std::arg(output[k]), exp_phase_2[k], ERR) << "at index: " << k;
    }

    // compute FFT of a more complex signal
    const std::vector<float> input_3 = {0, 6, -1, 3, 3, 0, -5, 2};
    const std::vector<float> exp_mag_3 = {8, 8.9657558, 9.0553851, 6.6041823, 14, 6.6041823, 9.0553851, 8.9657558};
    const std::vector<float> exp_phase_3 = {0*PI, -0.4809757*PI, -0.035223287*PI, -0.95406458*PI, 1*PI, 0.95406458*PI, 0.035223287*PI, 0.4809757*PI};

    EXPECT_FALSE(fft.transform(input_3, output));

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(std::abs(output[k]), exp_mag_3[k], ERR) << "at index: " << k;
        EXPECT_NEAR(std::arg(output[k]), exp_phase_3[k], ERR) << "at index: " << k;
    }
}

/**
 * FFT tests of forward transform for larger input
 */
TEST(DiscreteFourierTest, LargeTransforms) {
    // compute 16-point FFT
    const int size_1 = 16;
    FFT<double> fft_16(size_1);
    std::vector<std::complex<double>> output_1(size_1);

    const std::vector<double> input_1 = {0.19325677, 0.50802583, 0.43814404, 0.32752371, 0.95930142, 0.064225197, 0.3702718, 0.7806024, 0.92368416, 0.15440416,
        0.36631957, 0.6990894, 0.65803902, 0.76735618, 0.5950245, 0.49130181};
    const std::vector<double> exp_mag_1 = {8.29657, 0.64687173, 0.63673424, 1.3593968, 1.2559982, 1.5603724, 0.56979207, 1.1762005, 0.71151258, 1.1762005,
        0.56979207, 1.5603724, 1.2559982, 1.3593968, 0.63673424, 0.64687173};
    const std::vector<double> exp_phase_1 = {0*PI, 0.67293605*PI, 0.74740989*PI, -0.85557327*PI, 0.22128576*PI, -0.63866057*PI, 0.92545546*PI,
        0.7052744*PI, 0*PI, -0.7052744*PI, -0.92545546*PI, 0.63866057*PI, -0.22128576*PI, 0.85557327*PI, -0.74740989*PI, -0.67293605*PI};
    
    EXPECT_FALSE(fft_16.transform(input_1, output_1));

    for(int k = 0; k < size_1; k++) {
        EXPECT_NEAR(std::abs(output_1[k]), exp_mag_1[k], ERR) << "at index: " << k;
        EXPECT_NEAR(std::arg(output_1[k]), exp_phase_1[k], ERR) << "at index: " << k;
    }

    // compute 64-point FFT
    const int size_2 = 64;
    FFT<float> fft_64(size_2);
    std::vector<std::complex<float>> output_2(size_2);
    const std::vector<float> input_2 = {-3.2265482, -10.410971, -1.6029436, -2.3191998, -12.83878, -9.523311, -17.026635, -17.860913, -8.3971328, -1.7158779,
        -19.456333, -17.310326, -14.54653,-11.036449, -6.8549252, -9.1721527, -16.245687, -0.21167737, -13.692694, -9.4197281, -4.9385522, -18.57603,
        -13.649794, -10.015527, -17.539106, -11.495736, -7.9912412, -4.669628, -9.6095481, -18.020386, -7.670451, -10.762271, -7.6599028, -8.6756561,
        -0.66165614, -8.2098469, -3.7220132, -4.21991, -4.5209202, -12.830944, -9.816142, -3.8762089, -7.9045636, -18.61622, -8.6183354, -13.535054,
        -14.159279, -0.030572655, -16.047672, -16.375155, -16.174376, -3.3927848, -6.3534916, -7.6606568, -19.813324, -3.1395693, -18.306159, -19.769588,
        -9.8017529, -3.1108698, -12.550191, -10.690679, -13.415136, -2.89461};
    
    const std::vector<float> exp_mag_2 = {644.36033, 25.500682, 62.954245, 37.897821, 66.583686, 39.081913, 6.1653048, 36.541188, 35.446397, 26.162528,
        69.585455, 19.145259, 48.883553, 38.008732, 62.209237, 52.592144, 32.284477, 26.624314, 92.07092, 19.975108, 16.696912, 24.163459, 68.13828, 80.716006,
        39.335661, 29.794907, 36.728273, 11.338773, 8.4897161, 18.177091, 53.586623, 53.408066, 45.263307, 53.408066, 53.586623, 18.177091, 8.4897161, 11.338773,
        36.728273, 29.794907, 39.335661, 80.716006, 68.13828, 24.163459, 16.696912, 19.975108, 92.07092, 26.624314, 32.284477, 52.592144, 62.209237, 38.008732,
        48.883553, 19.145259, 69.585455, 26.162528, 35.446397, 36.541188, 6.1653048, 39.081913, 66.583686, 37.897821, 62.954245, 25.500682};
    
    const std::vector<float> exp_phase_2 = {1*PI, 0.78223009*PI, -0.12927332*PI, 0.33424481*PI, -0.21162325*PI, -0.11174637*PI, -0.99137164*PI, 0.22351888*PI,
        -0.58319747*PI, -0.71550259*PI, -0.48310445*PI, -0.52711311*PI, 0.53588553*PI, 0.79488062*PI, 0.073007935*PI, 0.070882116*PI, 0.46065664*PI, 0.6858931*PI,
        0.95395952*PI, -0.10913799*PI, 0.69076309*PI, 0.7397536*PI, 0.11450948*PI, 0.050580059*PI, 0.95610429*PI, 0.67828382*PI, 0.043766452*PI, 0.98954333*PI,
        -0.24858459*PI, -0.099937212*PI, 0.16835966*PI, 0.96433365*PI, 1*PI, -0.96433365*PI, -0.16835966*PI, 0.099937212*PI, 0.24858459*PI, -0.98954333*PI,
        -0.043766452*PI, -0.67828382*PI, -0.95610429*PI, -0.050580059*PI, -0.11450948*PI, -0.7397536*PI, -0.69076309*PI, 0.10913799*PI, -0.95395952*PI,
        -0.6858931*PI, -0.46065664*PI, -0.070882116*PI, -0.073007935*PI, -0.79488062*PI, -0.53588553*PI, 0.52711311*PI, 0.48310445*PI, 0.71550259*PI, 0.58319747*PI,
        -0.22351888*PI, 0.99137164*PI, 0.11174637*PI, 0.21162325*PI, -0.33424481*PI, 0.12927332*PI, -0.78223009*PI};

    EXPECT_FALSE(fft_64.transform(input_2, output_2));

    // increase error tolerance for larger signals with larger magnitude
    for(int k = 0; k < size_2; k++) {
        EXPECT_NEAR(std::abs(output_2[k]), exp_mag_2[k], 50 * ERR) << "at index: " << k;
        EXPECT_NEAR(std::arg(output_2[k]), exp_phase_2[k], 50 * ERR) << "at index: " << k;
    }


}

/**
 * FFT test for inverse of transform being equivalent to identity
 */
TEST(DiscreteFourierTest, IdentityTransforms) {
    // test 8-point FFT
    const int size_1 = 8;
    FFT<double> fft_8(size_1);
    const std::vector<double> input_1 = {-0.67734518, 0.041097089, -0.83970564, -0.76808002, 0.28387797, -0.047668902, 0.70266354, 0.78609125};

    std::vector<std::complex<double>> transformed = fft_8.transform(input_1);
    std::vector<double> output = fft_8.inverse(transformed);

    for(int k = 0; k < size_1; k++) {
        EXPECT_NEAR(input_1[k], output[k], ERR) << "at index: " << k;
    }

    

    // test 16-point FFT
    const int size_2 = 16;
    FFT<double> fft_16(size_2);
    const std::vector<double> input_2 = {-0.80143361, -4.1420611, 4.7281667, 3.655163, 2.2732652, 0.51822452, -1.5725374, -0.41392366, 3.1228435, -4.8941613, 1.8463468, -0.29013594, -2.5307239, 4.2880149, 1.8248971, 0.0077635051};

    transformed = fft_16.transform(input_2);
    output = fft_16.inverse(transformed);

    for(int k = 0; k < size_2; k++) {
        EXPECT_NEAR(input_2[k], output[k], ERR) << "at index: " << k;
    }
}

/**
 * Forward and inverse FFT test for lengths that are not a power of 2
 */
TEST(DiscreteFourierTest, ZeroPadding) {
    // test 6-point FFT
    const int size_1 = 6;
    const int eff_size_1 = 8;
    std::vector<std::complex<double>> transformed(eff_size_1);
    std::vector<double> output(eff_size_1);

    FFT<double> fft_6(size_1);
    const std::vector<double> input_1 = {-0.67734518, 0.041097089, -0.83970564, -0.76808002, 0.28387797, -0.047668902};

    EXPECT_EQ(fft_6.input_size, eff_size_1);
    EXPECT_FALSE(fft_6.transform(input_1, transformed));
    EXPECT_FALSE(fft_6.inverse(transformed, output));

    for(int k = 0; k < size_1; k++) {
        EXPECT_NEAR(input_1[k], output[k], ERR) << "at index: " << k;
    }
    for(int k = size_1; k < eff_size_1; k++) {
        EXPECT_NEAR(0, output[k], ERR) << "at index: " << k;
    }

    

    // test 17-point FFT
    const int size_2 = 17;
    const int eff_size_2 = 32;
    FFT<double> fft_17(size_2);
    const std::vector<double> input_2 = {-0.80143361, -4.1420611, 4.7281667, 3.655163, 2.2732652, 0.51822452, -1.5725374, 0, -0.41392366, 3.1228435, -4.8941613, 1.8463468, -0.29013594, -2.5307239, 4.2880149, 1.8248971, 0.0077635051};
    transformed.resize(eff_size_2);
    output.resize(eff_size_2);

    EXPECT_FALSE(fft_17.transform(input_2, transformed));
    EXPECT_FALSE(fft_17.inverse(transformed, output));

    for(int k = 0; k < size_2; k++) {
        EXPECT_NEAR(input_2[k], output[k], ERR) << "at index: " << k;
    }

    for(int k = size_2; k < eff_size_2; k++) {
        EXPECT_NEAR(0, output[k], ERR) << "at index: " << k;
    }
}