#ifndef __JAGGED_MCM_H__
#define __JAGGED_MCM_H__

#include <array>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <span>
#include <utility>


/**
 * TODO: "Jagged_Multichannel_Matrix" is a very long name that is annoying to type, find a better name.
 * Could switch Multichannel_Matrix to MCM or MC_Matrix but I was hoping to avoid too many abbreviations.
 * On the other hand, abbreviations are already used for FFT and DCT e.g. Discrete_Cosine_Transform_2 is very long
 */
#include "2d_transforms.hpp"

/**
 * Idea:
 * 
 * Have a Jagged_Multichannel_Matrix which contains a Matrix_View for each channel.
 * 
 * Matrix_View is the same as a single channel multichannel matrix but uses spans instead of shared pointers. The shared pointer
 * is owned by the wrapping Jagged_Multichannel_Matrix.
 * 
 * Multichannel_Matrix is already a template which takes its container as a parameter so perhaps Matrix_View could be created from
 * the same template or slightly modified version of the same template.
 */
template<Has_Arithmetic T, int Channels>
class MCM_View : public Abstract_Contiguous_MCM<T, Channels> {
private:
    using Abstract_MCM_t = Abstract_Contiguous_MCM<T, Channels>;
public:
    std::span<T> data;

    T& operator[] (int i) override {return (*this->data)[i];}
    const T& operator[] (int i) const override {return (*this->data)[i];}
    using Abstract_MCM_t::index;

/**
 * Constructors
 */
private:
    MCM_View(std::span<T> data, Shape shape, Strides strides, Order order)
        : Abstract_MCM_t(shape, strides, order), data(data) {}
public:
    MCM_View(std::span<T> data, Shape shape, Order order = Order::ROW_COL_CH)
        : Abstract_MCM_t(shape, order), data(data) {}

    MCM_View()
        : MCM_View(std::span<T>(), {0, 0}) {}


/** 
 * Data reformatting
 * 
 * TODO: See Multichannel_Matrix for discussion
 */ 
    MCM_View<T, Channels> reshape(Shape shape, bool inplace = false) {
        if(!inplace) {
            this->Abstract_MCM_t::reshape(shape);
            return *this;
        }
        else {
            // TODO: does the default move constructor work instead?
            // also worth considering if it is worth adding another constructor to avoid calculating strides twice here
            MCM_View<T, Channels> new_mat(*this);
            new_mat.Abstract_MCM_t::reshape(shape);
            return new_mat;
        }
    }

// see discussion in reshape(...) for more on design decisions
    MCM_View<T, Channels> transpose(bool inplace = false) {
        if(!inplace) {
            this->Abstract_MCM_t::transpose();
            return *this;
        }
        else {
            MCM_View<T, Channels> new_mat(*this);
            new_mat.Abstract_MCM_t::transpose();
            return new_mat;
        }
    }
};

template<Has_Arithmetic T>
using Matrix_View = MCM_View<T, 1>;

template<Has_Arithmetic T, int Channels>
class Jagged_Multichannel_Matrix {
public:
    std::shared_ptr<std::vector<T>> data;
    
private:
    std::array<Matrix_View<T>, Channels> submatrices;
    int size_;
public:
    Matrix_View<T> channel_view(int k) {
        if(k < Channels && k >= 0) {
            return this->submatrices[k];
        }
        else {
            // return empty matrix if not a valid index, bounds checking added because this function not called often
            return Matrix_View<T>();
        }
    }
    /**
     * Indexing method without bounds checks
     */
    T& index(int i, int j, int k) {
       return  this->submatrices[k].index(i, j);
    }

    Jagged_Multichannel_Matrix(std::initializer_list<T> data, std::initializer_list<Shape> shapes)
        : data(std::shared_ptr<std::vector<T>>(new std::vector<T>(data))) {
        
            // TODO: find something better than assert
        assert(shapes.size() == Channels);
        
        for(Shape shape: shapes) {
            this->size_ += shape.size();
        }

        // if initializer list was smaller than jagged matrix, extend with zeros
        if(this->data->size() <= this->size()) {
            this->data->resize(this->size());
        }

        auto left = this->data->begin();
        auto shape_iter = shapes.begin();
        for(int k = 0; k < Channels; k++) {
            auto right = left + shape_iter->size();
            this->submatrices[k] = Matrix_View<T>(std::span<T>(left, right), *shape_iter);
            left = right;
            shape_iter++;
        }

    }
    
    int size() {return this->size_;}
};



#endif
