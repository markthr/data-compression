#ifndef __IMAGE_FILE_IO_H__
#define __IMAGE_FILE_IO_H__

#include <string>
// TODO: is there a better way of handling this import?
#include "../transforms/2d/2d_transforms.hpp"

namespace img {
    /**
     * Returns an empty matrix if file could not be read
     * 
     * TODO: look into allowing the shape to be inferred
     */
    template<Has_Arithmetic T>
    Image_Matrix<T> read_img_csv(std::string path, Shape shape, Order order, char delim = ',');
}

#include "image_file_io_impl.hpp"

#endif