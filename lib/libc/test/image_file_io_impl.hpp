#ifndef __IMAGE_FILE_IO_IMPL_H__
#define __IMAGE_FILE_IO_IMPL_H__

#include <charconv>
#include <fstream>
#include "image_file_io.hpp"

template<Has_Arithmetic T>
Image_Matrix<T> img::read_img_csv(std::string path, Shape shape, Order order, char delim) {
    std::ifstream file("data/orioles_mascot_cropped.csv");

    if(!file.is_open()) {
        return Image_Matrix<T>();
    }

    Image_Matrix<T> image(shape, order);
    std::string line;
    if(file.is_open()) {
        int n = 0;
        while(std::getline(file, line)) {
            auto left = line.begin();
            auto right = left + 1; // assuming line is not empty
            while(left != line.end()) {
                if(right == line.end() || *right == delim) {
                    double val;
                    std::from_chars(&(*left), &(*right), val);
                    image.index(n++) = val;
                    
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
    }
    file.close();
    return image;
}

#endif