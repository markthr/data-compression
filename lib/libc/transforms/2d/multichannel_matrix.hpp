#ifndef __MULTICHANNEL_MATRIX_H__
#define __MULTICHANNEL_MATRIX_H__

#include "2d_transforms.hpp"
#include <memory>
#include <utility>
#include <span>
#include <initializer_list>

struct Shape {
    int m; // m=height=# of rows
    int n; // n=width=# of cols
};

enum class Dimension {
    COLUMN,
    ROW,
    CHANNEL
};

struct Order {
    Dimension first;
    Dimension second;
    Dimension third;
    
    static const Order ROW_COL_CH;
    static const Order CH_ROW_COL;
};

constexpr Order Order::ROW_COL_CH = {Dimension::ROW, Dimension::COLUMN, Dimension::CHANNEL};
constexpr Order Order::CH_ROW_COL = {Dimension::CHANNEL, Dimension::ROW, Dimension::COLUMN};



/**
 * Represents the number of offsets in the underlying array-like data to traverse to the next element along a dimension
 * 
 * Strides.row: index(i,j,k) -> index(i+1, j, k)
 * Strides.col: index(i,j,k) -> index(i, j+1, k)
 * Strides.ch: index(i,j,k) -> index(i, j, k+1)
 * 
 * Names refer to the dimension to travel along (e.g. increasing rows), not the dimension to travel within (e.g. fixed row)
 * 
 * TODO: is this good naming? can the naming be improved?
 */
struct Strides {
    int row;
    int col;
    int ch;
};

template<typename T, int Channels, std::size_t Extent = std::dynamic_extent>
class Multichannel_Matrix {
private:
    /**
     * Used to compute the strides for a matrix.
     * This is used each time Shape is set either in a constructor or via reshape(...) and the result is saved in the strides_ field.
     */
    static Strides compute_strides(Shape shape, Order order);

    static Order swap_row_col(Order order);

    // following the convention of using _ as a suffix for internal variables
    std::shared_ptr<std::vector<T>> data;
    Shape shape_;
    Order order_;
    Strides strides_;
    int size_;
    
    /**
     * Internal constructor, not currently intended to be able to manually specify strides. Strides should be set based
     * on the passed Order struct.
     */
    Multichannel_Matrix(std::shared_ptr<std::vector<T>> data, Shape shape, Strides strides, Order order = Order::ROW_COL_CH)
        : data(data), shape_(shape), order_(order), size_(shape.m*shape.n*Channels), strides_(strides) {}
public:
// Constructors

    /**
     * Constructor for initializing data based an an initializer list. A std::initializer_list is used in order to be able to pass the
     * list through functions.
     */
    Multichannel_Matrix(std::initializer_list<T> data, Shape shape, Order order =  Order::ROW_COL_CH)
        : Multichannel_Matrix(std::shared_ptr<std::vector<T>>(new std::vector<T>(data)), shape, compute_strides(shape, order), order) {

        if((this->data)->size() != this->size()) {
            (this->data)->resize(this->size());
        }
    }
    
    /**
     * Constructor for creating a matrix of all zeros.
     */
    Multichannel_Matrix(Shape shape, Order order =  Order::ROW_COL_CH)
        : Multichannel_Matrix(std::shared_ptr<std::vector<T>>(new std::vector<T>(shape.m*shape.n*Channels)), shape, compute_strides(shape, order), order) {}
    
    /**
     * Shallow copy constructor
     */
    Multichannel_Matrix(const Multichannel_Matrix<T, Channels, Extent>& other) 
        : Multichannel_Matrix(other.data, other.shape_, other.strides_, other.order_) {}
    
// Getters
    const Shape& shape() const {return this->shape_;}
    const Strides& strides() const {return this->strides_;}
    Order order() const {return this->order_;}
    int size() const {return this->size_;} // TODO: should size be m*n or m*n*channels    

    /**
     * Adding an empty constructor for dynamic extent matrices without full template specialization
     * Perhaps there is a way to avoid this with CRTP, but CRTP seems like it'd be overkill
     * 
     * TODO: currently static extent matrices (able to live entirely on the stack) are Multichannel_Matrixnot supported.
     * Is it worth adding support or should extent be removed from the template parameters.
     */
    Multichannel_Matrix() :Multichannel_Matrix({}, 0, 0){
        // allow dynamic extent matrix views to be trivially constructable.
        static_assert(Extent == std::dynamic_extent, "Only dynamic extent matrices are trivially constructable");
    }

// Indexing methods
// TODO: is there a better name for this operator? Is there a way to avoid having to write both const and non const version?
    T& index(int i, int j, int k) {
        return (*this->data)[i * this->strides_.row + j * this->strides_.col + k * this->strides_.ch];
    }
    const T& index(int i, int j, int k) const {
        return (*this->data)[i * this->strides_.row + j * this->strides_.col + k * this->strides_.ch];
    }

    // only enabled for Channels=1
    T& index(int i, int j) {
        static_assert(Channels == 1);
        return (*this->data)[i * this->strides_.row + j * this->strides_.col];
    }
    const T& index(int i, int j) const {
        static_assert(Channels == 1);
        return (*this->data)[i * this->strides_.row + j * this->strides_.col];
    }

    // direct index on underlying contiguous memory, perhaps use [] instead here or add it and support both?
    T& index(int i) {return (*this->data)[i];}
    const T& index(int i) const {return (*this->data)[i];}

// Data reformatting
    Multichannel_Matrix<T, Channels, Extent> reshape(Shape shape, bool inplace = false) {
        assert(this->shape().m * this->shape().n == shape.m * shape.n); // ensure size does not change

        if(inplace) {
            this->shape_ = shape;
            this->strides_ = compute_strides(this->shape_, this->order_);
            return *this;
        }
        else {
            // TODO: does the default move constructor work instead?
            // also worth considering if it is worth adding another constructor to avoid calculating strides twice here
            Multichannel_Matrix<T, Channels, Extent> new_mat(*this);
            new_mat.shape_ = shape;
            new_mat.strides_ = compute_strides(new_mat.shape_, new_mat.order_);
            return new_mat;
        }
    }

    Multichannel_Matrix<T, Channels, Extent> transpose(bool inplace = false) {
        if(inplace) {
            this->shape_ = {this->shape_.n, this->shape_.m}; // swap components of Shape
            this->order_ = swap_row_col(this->order_);
            this->strides_ = compute_strides(this->shape_, this->order_);
            return *this;
        }
        else {
            // TODO: does the default move constructor work instead?
            Multichannel_Matrix<T, Channels, Extent> new_mat(*this);
            new_mat.shape_ = {this->shape_.n, this->shape_.m};
            new_mat.order_ = swap_row_col(this->order_);
            new_mat.strides_ = compute_strides(new_mat.shape_, new_mat.order_);
            
            return new_mat;
        }
    }
};

// TODO: perhaps follow the _t convention for typedef or does using have a different convention?
// convenience alias for single channel matrices
template<typename T, std::size_t Extent = std::dynamic_extent>
using Matrix = Multichannel_Matrix<T, 1, Extent>;

// convenience alias for 3 channel matrices
template<typename T, std::size_t Extent = std::dynamic_extent>
using Image_Matrix = Multichannel_Matrix<T, 3, Extent>;





// template<typename T, int Channels, std::size_t Extent>
// Matrix<T> Multichannel_Matrix<T, Channels, Extent>::channel(int k) {
//     return Multichannel_Matrix<T, 1>((*this->data).subspan(this->strides_.ch * k, this->strides_.ch));
// }



template<typename T, int Channels, std::size_t Extent>
Strides Multichannel_Matrix<T, Channels, Extent>::compute_strides(Shape shape, Order order) {
    // ensure dimensions are not repeated in the ordering
    assert(order.first != order.second && order.first != order.third && order.second != order.third); 
    Strides strides;
    
    int multiplier = 1;

    const int n_dim = 3;
    Dimension dims[n_dim] = {order.first, order.second, order.third};
    for(int d = 0; d < n_dim; d++) {
        if(dims[d] == Dimension::ROW) {
            strides.col = multiplier;
            multiplier *= shape.n;
        } else if(dims[d] == Dimension::COLUMN) {
            strides.row = multiplier;
            multiplier *= shape.m;
        }
        else { // must be Dimension::Channel
            strides.ch = multiplier;
            multiplier = Channels;
        }
    }

    return strides;
}

/**
 * Swaps the order of row and col so an order of {Row, Channel, Column} yields {Column, Channel, Row}
 * 
 * It is assumed that a valid order is supplied
 */
template<typename T, int Channels, std::size_t Extent>
Order Multichannel_Matrix<T, Channels, Extent>::swap_row_col(Order order) {

    const int n_dim = 3;
    Dimension dims[n_dim] = {order.first, order.second, order.third};

    int row_pos;
    int col_pos;

    for(int i = 0; i < n_dim; i++) {
        if(dims[i] == Dimension::ROW) {
            row_pos = i;
        }
        else if(dims[i] == Dimension::COLUMN) {
            col_pos = i;
        }
    }

    dims[row_pos] = Dimension::COLUMN;
    dims[col_pos] = Dimension::ROW;

    // could I forward the array instead?
    return {dims[0], dims[1], dims[2]};
}

#endif