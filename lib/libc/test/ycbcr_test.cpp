#include <gtest/gtest.h>
#include <array>
#include <fstream>
#include <string>
#include <string_view>
#include <charconv>
#include <sstream>


// TODO: is there a better way of handling this import?
#include "../transforms/2d/2d_transforms.hpp"

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
    std::array<float, size> exp_identity  = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    for(int k = 0; k < size; k++) {
        EXPECT_NEAR(exp_identity[k], identity[k], ERR) << "Coeffs not equal at index: " << k;
    }

    const int CHANNELS = 3;
    std::array<double, CHANNELS> exp_product = {191.979, -16.9182, 22.1262};
    std::array<double, CHANNELS> pixel_data = {223, 182, 162};
    Matrix<double, CHANNELS> pixel_mat(std::span<double, CHANNELS>(pixel_data), {CHANNELS, 1});
    std::vector<double> product = mat::multiply(ytr.transform_matrix, pixel_mat);

    for(int k = 0; k < CHANNELS; k++) {
        exp_product[k]++;
        product[k]++;
        EXPECT_NEAR(exp_product[k], product[k], ERR) << "Product not equal at index: " << k;
    }
}

TEST(YCbCrTest, IdentityTransforms) {
    const float ERR = 1e-4;
    const char DELIM = ',';
    const Shape SHAPE{327, 223};
    const int CHANNELS = 3;
    const int SIZE = SHAPE.m * SHAPE.n * CHANNELS;
    std::ifstream file("data/orioles_mascot_cropped.csv");
    EXPECT_TRUE(file.is_open()) << "Unable to locate test data";

    // TODO: package the CSV parsing into a function and make the data available across tests
    std::vector<double> input_data;
    input_data.reserve(SIZE);
    std::string line;
    if(file.is_open()) {
        while(std::getline(file, line)) {
            auto left = line.begin();
            auto right = left + 1; // assuming line is not empty
            while(left != line.end()) {
                if(right == line.end() || *right == DELIM) {
                    double val;
                    std::string_view sv(left, right); 
                    std::from_chars(sv.begin(), sv.end(), val);
                    input_data.push_back(val);
                    
                    if(right != line.end()) {
                        // skip past delim
                        left = ++right;
                    }
                    else {
                        // set exit condition
                        left = right;
                    }
                }
                
                if(right != line.end()) {
                    right++;
                }
            }
        }

        EXPECT_EQ(input_data.size(), SIZE) << "Failed to parse test data";
        file.close();

        std::vector<double> transformed_data(SIZE);
        std::vector<double> output_data(SIZE);
        
        Image_View<double> input(input_data, SHAPE, Order::CH_ROW_COL);
        Image_View<double> transformed(transformed_data, SHAPE, Order::CH_ROW_COL);
        Image_View<double> output(output_data, SHAPE, Order::CH_ROW_COL);

        YCbCr_Transformer<double> ycbcr(SHAPE);
        EXPECT_FALSE(ycbcr.transform(input, transformed)) << "Forward transform failed";

        for(int i = 0; i < input.shape().m && i < input.shape().n; i++) {
            for(int k = 0; k < CHANNELS; k++) {
                if(input.index(i, i, k)) {
                    EXPECT_NE(input.index(i, i, k), transformed.index(i, i, k));
                }
            }
        }

        EXPECT_FALSE(ycbcr.inverse(transformed, output))  << "Inverse transform failed";

        for(int n = 0; n < input.size(); n++) {
            if(std::abs(input.index(n) - output.index(n)) > ERR) {
                EXPECT_NEAR(input.index(n), output.index(n), ERR) << "Transformed value not equal at index: " << n;
                break;
            }
        }
    }
}