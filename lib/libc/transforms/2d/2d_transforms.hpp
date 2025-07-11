#ifndef __2D_TRANSFORMS_H__
#define __2D_TRANSFORMS_H__

#include <span>
#include <array>
#include <memory>
#include <utility>
#include <initializer_list>
#include "../transforms.hpp"

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

const Order Order::ROW_COL_CH = {Dimension::ROW, Dimension::COLUMN, Dimension::CHANNEL};
const Order Order::CH_ROW_COL = {Dimension::CHANNEL, Dimension::ROW, Dimension::COLUMN};



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

/**
 * Abstract type for fixed size invertible transforms that convert a floating point sequence
 * of up to length n to some domain specified by U
 */
template<std::floating_point T, typename U, template<typename> class Container = std::span>
class Abstract_Transformer_NC {
    public:
        const std::size_t input_size;
        const std::size_t output_size;

        Abstract_Transformer_NC(std::size_t input_size, std::size_t output_size) : input_size(input_size), output_size(output_size) {}

        Abstract_Transformer_NC(std::size_t size) : Abstract_Transformer_NC(size, size) {}

        /**
         *  Return 0 if successful, -1 otherwise
         */
        virtual int transform(Container<T> in, Container<U> out) = 0;
        virtual int inverse(Container<U> in, Container<T> out) = 0;
        

        std::vector<U> transform(Container<const T> in);
        std::vector<T> inverse(Container<const U> in);
    
    // helper methods, a new copy is made for every parameterization of the template
    // perhaps it could eventually be worthwhile to make static versions that exist outside of templates
    protected:
        /**
         * Utility that finds the least multiple of 2 greater than n
         */
        static int radix_2_size(int n);

        /**
         * Utility function for transfering input indices to output indices
         */
        static int bit_reversal(int bits, int num_bits);
};

// template<typename T>
// typedef shared_vec<T> std::shared_ptr<std::vector<T>>

template<typename T, int Channels, std::size_t Extent = std::dynamic_extent>
class Multichannel_Matrix {
    public:
        


    private:
        static Strides compute_strides(Shape shape, Order order);

        // following the convention of using _ as a suffix for internal variables
        std::shared_ptr<std::vector<T>> data;
        Shape shape_;
        Order order_;
        Strides strides_;
        int size_;

        Multichannel_Matrix(std::shared_ptr<std::vector<T>> data, Shape shape, Strides strides, Order order = Order::ROW_COL_CH);
    public:
        // declaring getters here, because the verbosity of moving getters to the impl file seems excessive
        const Shape& shape() const {return this->shape_;}
        const Strides& strides() const {return this->strides_;}
        Order order() const {return this->order_;}
        int size() const {return this->size_;} // TODO: should size be m*n or m*n*channels

        
        Multichannel_Matrix(std::initializer_list<T> data, Shape shape, Order order =  Order::ROW_COL_CH);
        Multichannel_Matrix(Shape shape, Order order =  Order::ROW_COL_CH);
        
        void reshape(Shape shape);
        

        /**
         * Adding an empty constructor for dynamic extent views without full template specialization
         * Perhaps there is a way to avoid this with CRTP, but CRTP seems like it'd be overkill
         */
        Multichannel_Matrix() :Multichannel_Matrix({}, 0, 0){
            // allow dynamic extent matrix views to be trivially constructable.
            static_assert(Extent == std::dynamic_extent, "Only dynamic extent matrices are trivially constructable");
        }

        // TODO: is there a better name for this operator? Is there a way to avoid having to write both const and non const version?
        T& index(int i, int j, int k);
        const T& index(int i, int j, int k) const;
        // only enabled for Channels=1
        T& index(int i, int j);
        const T& index(int i, int j) const;
        // direct index on underlying contiguous memory, perhaps use [] instead here?
        T& index(int i);
        const T& index(int i) const;

        // Multichannel_Matrix<T, 1> channel(int k);
};

template<typename T, std::size_t Extent = std::dynamic_extent>
using Matrix = Multichannel_Matrix<T, 1, Extent>;

template<typename T, std::size_t Extent = std::dynamic_extent>
using Image_View = Multichannel_Matrix<T, 3, Extent>;

// needs to be declared before ycbcr transformer
#include "matrix_operations_impl.hpp"

template<typename T, typename U>
class Abstract_Image_Transformer : public Abstract_Transformer_NC<T, U, Image_View> {
    
    public:
        const Shape input_shape;
        const Shape output_shape;

        Abstract_Image_Transformer(const Shape shape)
            : input_shape(shape), output_shape(shape), 
            Abstract_Transformer_NC<T, U, Image_View>(
                shape.m * shape.n * 3) {}
        
        Abstract_Image_Transformer(const Shape input_shape, const Shape output_shape)
            : input_shape(input_shape), output_shape(output_shape),
            Abstract_Transformer_NC<T, U, Image_View>(
                input_shape.m * input_shape.n * 3, output_shape.m * output_shape.n * 3) {}

};







template<typename T>
class YCbCr_Transformer : public Abstract_Image_Transformer<T, T> {
    public:
        static const int Channels = 3;
    public:
        const float k_r;
        const float k_g;
        const float k_b;
        Multichannel_Matrix<T, 1> transform_matrix;
        Multichannel_Matrix<T, 1> inverse_matrix;

        /**
         * Default values for k_b and k_r are taken from ITU-R BT.601
         * 
         * k_g is determined by the formula k_b + k_r + k_g = 1
         * 
         * There are thus ways that bad values can be provided e.g. making one of the coefficients zero
         */
        YCbCr_Transformer(const Shape shape, float k_b = 0.114, float k_r = 0.299);
        // TODO: decide how to handle bad parameters in the constructor e.g. exception, public method which wraps a private constructor, etc
        
        // indicate override of pure virtual signature
        int transform(Image_View<T> in, Image_View<T> out) override;
        int inverse(Image_View<T> in, Image_View<T> out) override;

        /**
         * Use the current values for k_r, k_g, and k_b to compute and set forward transform matrix
         */
        void compute_forward_transform();

        /**
         * Use the current values for k_r, k_g, and k_b to compute and set forward inverse matrix
         */
        void compute_inverse_transform();
};



#include "multichannel_matrix_impl.hpp"
#include "ycbcr_transformer_impl.hpp"
#include "bitdepth_transformer_impl.hpp"

#endif