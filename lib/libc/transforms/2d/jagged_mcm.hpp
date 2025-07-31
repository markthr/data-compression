#ifndef __JAGGED_MCM_H__
#define __JAGGED_MCM_H__

#include <array>
#include <cassert>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>


/**
 * TODO: "Jagged_Multichannel_Matrix" is a very long name that is annoying to type, find a better name.
 * Could switch Multichannel_Matrix to MCM or MC_Matrix but I was hoping to avoid too many abbreviations.
 * On the other hand, abbreviations are already used for FFT and DCT e.g. Discrete_Cosine_Transform_2 is very long
 */
#include "2d_transforms.hpp"


/**
 * Concept used for Jagged matrix constructor
 * 
 * TODO: the iterator is copied, does input_range guarantee support for that?
 */
template<typename R, typename T, std::size_t Size>
concept Input_Range_Of = requires(R r) {
    std::ranges::input_range<R>;
    std::is_same<std::ranges::range_value_t<R>, T>::value;
    r.size() == Size; // TODO: is this actually enforcing compile time const size? Is enforcing that desirable?
};

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

    T& operator[] (int i) override {return this->data[i];}
    const T& operator[] (int i) const override {return this->data[i];}
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
    int size_ = 0;

    static std::array<Shape, Channels> shape_per_channel(Shape shape) {
        std::array<Shape, Channels> shapes;
        shapes.fill(shape);
        return shapes;
    }

    template<Input_Range_Of<Shape, Channels> Shape_Range>
    static int sum_shapes(Shape_Range shapes) {
        // TODO: use something better than assert. Should strict equality be required for the check?
        assert(shapes.size() >= Channels);
        int size = 0;
        for(auto shape: shapes) {
            size += shape.size();
        }
        return size;
    }

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

/**
 * Constructors and converters
 */
private:
    /**
     * Precondition: assumes size has been set
     * 
     * Input_It shapes: Iterator have a number of elements at least equal to Channels
     * 
     * TODO: should a check be added for the iterator ending here or just let bad code break?
     */
    template<Input_Range_Of<Shape, Channels> Shape_Range>
    Jagged_Multichannel_Matrix(std::shared_ptr<std::vector<T>> data, Shape_Range shapes)
        : data(data), size_(sum_shapes(shapes)) {
        
        // if initializer list was smaller than jagged matrix, extend with zeros
        if(this->data->size() <= this->size()) {
            this->data->resize(this->size());
        }

        auto left = this->data->begin();
        auto shape_it = shapes.begin();
        for(int k = 0; k < Channels; k++) {
            auto right = left + shape_it->size();
            this->submatrices[k] = Matrix_View<T>(std::span<T>(left, right), *shape_it);
            left = right;
            shape_it++;
        }
    }
public:
    Jagged_Multichannel_Matrix(std::initializer_list<T> data, std::initializer_list<Shape> shapes)
        : Jagged_Multichannel_Matrix(std::shared_ptr<std::vector<T>>(new std::vector<T>(data)), shapes) {}


    Jagged_Multichannel_Matrix(std::initializer_list<T> data, Shape shape)
        : Jagged_Multichannel_Matrix(std::shared_ptr<std::vector<T>>(new std::vector<T>(data)), shape_per_channel(shape)) {}

    Jagged_Multichannel_Matrix() {}

    // TODO: need to add a constructor to support this function
    template<Matrix_Like<T> M>
    static Jagged_Multichannel_Matrix<T, Channels> as_jagged(M& m) {
        
        // TODO: Should a different guard be used instead of assert?
        assert(m.shape().size() * Channels == m.size());

        return Jagged_Multichannel_Matrix<T, Channels>(m.data, shape_per_channel(m.shape()));
    }
    
    int size() {return this->size_;}
};



#endif
