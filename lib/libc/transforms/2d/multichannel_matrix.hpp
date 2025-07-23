#ifndef __MULTICHANNEL_MATRIX_H__
#define __MULTICHANNEL_MATRIX_H__

#include "2d_transforms.hpp"
#include <memory>
#include <utility>
#include <initializer_list>
#include <cassert>
#include <type_traits>

struct Shape {
    int m; // m=height=# of rows
    int n; // n=width=# of cols

    inline int size() const {return m * n;}
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

template<typename From, typename To>
struct Propagate_Const {typedef To type;};
template<typename From, typename To>
struct Propagate_Const<const From, To> {typedef std::add_const<To>::type type;};
template<typename From, typename To>
using Propagate_Const_t = Propagate_Const<From, To>::type;

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
 * A concept for testing if the given type pointer can be assigned from a std::share_ptr<Underlying>
 */
template<typename Pointer, typename Underlying>
concept Can_Share = requires {std::same_as<std::remove_cvref_t<Pointer>, std::shared_ptr<Underlying>>;}
    || requires {std::same_as<std::remove_cvref_t<Pointer>, std::shared_ptr<const Underlying>>;};



template<typename M, typename T>
concept Matrix_Like = requires(M m) {
    {m.data} -> Can_Share<T>;
    {m.shape()} -> std::same_as<Shape>;
    {m.order()} -> std::same_as<Order>;
    {m.strides()} -> std::same_as<Strides>;
    {m.size()} -> std::same_as<int>;
    Has_Arithmetic<T>;
};

template<Has_Arithmetic T, int Channels>
class Abstract_Contiguous_MCM {
private:
    // following the convention of using _ as a suffix for internal variables
    Shape shape_;
    Order order_;
    Strides strides_;
    int size_; // equals shape.n*shape.m*Channels


protected:
    /**
     * Internal constructor that other constructors delegate to
     * 
     * TODO: Should the assert be switched to a throw?
     */
    Abstract_Contiguous_MCM(Shape shape, Strides strides, Order order)
        : shape_(shape), order_(order), size_(shape.size() * Channels), strides_(strides) {}

    Abstract_Contiguous_MCM(Shape shape, Order order)
        : Abstract_Contiguous_MCM<T, Channels>(shape, compute_strides(shape, order), order) {}
    
    /**
     * Make trivially constructable
     */
    Abstract_Contiguous_MCM() {}
    
    /**
     * Used to compute the strides for a matrix.
     * This is used each time Shape is set either in a constructor or via reshape(...) and the result is saved in the strides_ field.
     */
    static Strides compute_strides(Shape shape, Order order);
    static Order swap_row_col(Order order);

public:
    // Getters
    Shape shape() const {return this->shape_;}
    Strides strides() const {return this->strides_;}
    Order order() const {return this->order_;}
    int size() const {return this->size_;}

    /**
     * Data reformatting methods
     * 
     * TODO: is there a better name for these? is there a better way to handle data copying?
     * 
     * TODO: it is desirable to have a version that returns a new matrix and an inplace version
     * Is there a way to have the version which returns a new matrix in the abstract class that isn't terrible?
     */
    void reshape(Shape shape) {
        // TODO: keep assert or switch to throwing?
        assert(this->shape().size() == shape.size()); // ensure size does not change

        this->shape_ = shape;
        this->strides_ = compute_strides(this->shape_, this->order_);
    }

    void transpose() {
        this->shape_ = {this->shape_.n, this->shape_.m}; // swap components of Shape
        this->order_ = swap_row_col(this->order_);
        this->strides_ = compute_strides(this->shape_, this->order_);
    }
    /**
     * Data access methods
     * 
     * Currently using static_assert instead of full template specialization to avoid identical declarations
     * TODO: is there a better alternative?
     * 
     * Pure virtual operator[] methods must be overridden by inheriting classes. Other data access methods delegate access to
     * the pure virtual methods.
     * 
     * Declared as operator[] instead of an overload of index to avoid having to manually include a using statement
     * in inheriting classes.
     */
    virtual const T& operator[] (int i) const = 0;
    virtual T& operator[] (int i) = 0; // TODO: this does not work

    // 3 index access for channels=1 allows a function to iterate over data across any number of channels (greater than 0) the same
    const T& index(int i, int j, int k) const {
        return this->operator[](i * this->strides_.row + j * this->strides_.col + k * this->strides_.ch);
    }

    T& index(int i, int j, int k) {
        return this->operator[](i * this->strides().row + j * this->strides().col + k * this->strides().ch);
    }


    /**
     * only enabled for Channels=1
     * 
     * TODO: switch to using a better method of having conditional member functions
     * https://brevzin.github.io/c++/2021/11/21/conditional-members/
     * 
     */
    const T& index(int i, int j) const {
        static_assert(Channels == 1);
        return this->operator[](i * this->strides_.row + j * this->strides_.col);
    }

    // only enabled for Channels=1
    T& index(int i, int j) {
        static_assert(Channels == 1);
        return this->operator[](i * this->strides().row + j * this->strides().col);
    }

    // TODO: perhaps also include at(...) variant with bounds checking
};

template<Has_Arithmetic T, int Channels, template<typename> typename Container = std::vector>
class Multichannel_Matrix : public Abstract_Contiguous_MCM<Propagate_Const_t<Container<T>, T>, Channels>{
private:
    using T_CV = Propagate_Const_t<Container<T>, T>;
    using Abstract_MCM_t = Abstract_Contiguous_MCM<T_CV, Channels>;

public:
    // declared public so concepts can access it because concepts cannot be declare friends so no private access
    std::shared_ptr<Container<T>> data;
private:

    /**
     * Internal constructor, not currently intended to be able to manually specify strides. Strides should be set based
     * on the passed Order struct.
     * 
     * If data.size() does not equal shape.m*shape.n * Channels then the underlying vector will be resized
     */
    Multichannel_Matrix(std::shared_ptr<Container<T>> data, Shape shape, Order order)
        : Abstract_MCM_t(shape, order), data(data) {}

    Multichannel_Matrix(std::shared_ptr<Container<T>> data, Shape shape, Strides strides, Order order)
        : Abstract_MCM_t(shape, strides, order), data(data) {}
public:
// Constructors

    /**
     * Constructor for initializing data based an an initializer list. A std::initializer_list is used in order to be able to pass the
     * list through functions.
     */
     Multichannel_Matrix(std::initializer_list<T> data, Shape shape, Order order =  Order::ROW_COL_CH)
        : Multichannel_Matrix(std::shared_ptr<Container<T>>(new Container<T>(data)), shape, order) {
        // extend data if initializer list was smaller than matrix size
        if(this->data->size() <= this->size()) {
            this->data->resize(this->size());
        }
    }
    
    /**
     * Constructor for creating a matrix of all zeros.
     */
    Multichannel_Matrix(Shape shape, Order order =  Order::ROW_COL_CH)
        : Multichannel_Matrix(std::shared_ptr<Container<T>>(new std::vector<T>(shape.m*shape.n*Channels)), shape, order) {}
    
    /**
     * Shallow copy constructor
     */
    Multichannel_Matrix(const Multichannel_Matrix<T, Channels, Container>& other) 
        : Multichannel_Matrix(other.data, other.shape(), other.strides(), other.order()) {}

    /**
     * Converter
     * TODO: is there a better name to describe what this does? Should this be private or protected?
     */
    template<Matrix_Like<T> M>
    static Multichannel_Matrix<T, Channels, Container> as_matrix(M& m) {
        
        // TODO: should this constructor be private? Should a different guard be used instead of assert?
        assert(m.shape().m * m.shape().n * Channels == m.size());

        return Multichannel_Matrix<T, Channels, Container>(m.data, m.shape(), m.strides(), m.order());
    }

    /**
     * Creates an empty matrix that is considered 0 x 0 
     * 
     * Trivially constructable matrices are usefule because they allow a matrix variable to be declared and then assigned to later
     */
    Multichannel_Matrix() :Multichannel_Matrix({}, {0, 0}){}

// Indexing methods
// TODO: is there a better name for this operator? Is there a way to avoid having to write both const and non const version?


    // direct index on underlying contiguous memory, perhaps use [] instead here or add it and support both?
    T_CV& operator[] (int i) override {return (*this->data)[i];}
    const T_CV& operator[] (int i) const override {return (*this->data)[i];}
    using Abstract_MCM_t::index;

    /** 
     * Data reformatting
     * 
     * TODO: should overloading be used instead so that transpose/reshape can be called on const matrices
     * Is there a "good" way to avoid writing the out of place version every time Abstract_Contiguous_MCM
     * gets extended?
     * Moving definition to parent class would have return type be a pointer or smart pointer because
     * abstract types cannot be returned by value. Maybe CRTP or other templating would solve this but does
     * that add needless complexity?
     */ 
    Multichannel_Matrix<T, Channels, Container> reshape(Shape shape, bool inplace = false) {
        if(!inplace) {
            this->Abstract_MCM_t::reshape(shape);
            return *this;
        }
        else {
            // TODO: does the default move constructor work instead?
            // also worth considering if it is worth adding another constructor to avoid calculating strides twice here
            Multichannel_Matrix<T, Channels, Container> new_mat(*this);
            new_mat.Abstract_MCM_t::reshape(shape);
            return new_mat;
        }
    }

    // see discussion in reshape(...) for more on design decisions
    Multichannel_Matrix<T, Channels, Container> transpose(bool inplace = false) {
    if(!inplace) {
            this->Abstract_MCM_t::transpose();
            return *this;
        }
        else {
            Multichannel_Matrix<T, Channels, Container> new_mat(*this);
            new_mat.Abstract_MCM_t::transpose();
            return new_mat;
        }
    }
};

// TODO: perhaps follow the _t convention for typedef or does using have a different convention?
// convenience alias for single channel matrices
template<Has_Arithmetic T, template<typename> class Container = std::vector>
using Matrix = Multichannel_Matrix<T, 1, Container>;

// convenience alias for 3 channel matrices
template<Has_Arithmetic T, template<typename> class Container = std::vector>
using Image_Matrix = Multichannel_Matrix<T, 3, Container>;

/**
 * 
 */

template<Has_Arithmetic T, int Channels>
Strides Abstract_Contiguous_MCM<T, Channels>::compute_strides(Shape shape, Order order) {
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
template<Has_Arithmetic T, int Channels>
Order Abstract_Contiguous_MCM<T, Channels>::swap_row_col(Order order) {

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

#include "jagged_mcm.hpp"

#endif