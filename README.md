# Data Compression

The goal of this project is to implement data compression algorithms using only the C++ standard library.
These compression algorithms and many of the transforms necessary to create them will be made available as a Python library using pybind11.

Currently, focus is on MPEG style image compression. MP3 style audio compression is an eventual goal.

## Base Transformations

### FFT: Fast Fourier Transform

Currently an iterative radix-2 FFT is implemented.

**Future Work:**

1. Switch to a recursive implementation to ensure cache coherance. It is worth considering whether it should be tail recursive.
The kernel of the recursive FFT should be a small iterative FFT e.g. size 8.

2. Add more FFT radixes e.g. radix-3, radix-5, and radix-7. The current implementation zero pads inputs to ensure they are always a power of 2.
In the worst case of $2^k+1$, this doubles the amount of computation. Supporting more FFT radixes would allow more flexible factorization which
improves the worst case.

3. Expand this FFT write up.

### DCT-II: Discrete Fourier Transform

Currently a type II DCT is implemented based on the implemented FFT discussed in the previous section.

**Future Work:**

1. Take advantage of symmetry. There is no need to perform a size $2n$ FFT when $n$ has even length.

2. Implement other DCT variants. Implementing type I, III, and IV DCTs would allow for empirical comparisons of how well each works in
different data compression algorithms.

3. Implement the modified discrete cosine transform (MDCT) which is a lapped version of DCT-IV. The overlap between consecutive blocks
makes the MDCT attractive for avoiding artifacts introduced by the boundary between blocks.

4. Expand this DCT write up.

## Transform Design

### Enforcing const correctness
 
A transform does not modify its input when generating an output. To enforce this, the C++ types containing the input are `const` qualified. The
input types `WrappingType<T>` used have reference semantics so `const WrappingType<T>` is shallow const so elements within a 
`const Container<T>` can still be modified. If the wrapping type is a `std::span`, then `std::span<const T>` will provide deep const, but trying
to convert `std::span<T>` to `std::span<const T>` will cause template substitution to fail. This issue is discussed at greater length
[in this blog](https://brevzin.github.io/c++/2021/09/10/deep-const/).

To address this, a public template function is used to accept a potentially deep const input and then coerce it into being deep const. The 
coerced value is then passed to a private pure virtual function which performs the actual transform. A const reference is used in order to
ensure the C++ compiler selects the `const` version of an overloaded functions. More detail is provided in the discussion of abstract transform
types.

### Common Concepts

#### `Has_Arithmetic`

The `Has_Arithmetic` concept abstracts types which support the 4 basic arithmetic operations. This concept is designed to accept 
`std::complex<...>` types in addition to the floating point and integer types that satisfy `std::is_arithmetic_v<T>`.

```cpp
template<typename T>
concept Has_Arithmetic = requires(T t1, T t2) {
    {t1 + t2} -> std::same_as<std::remove_cv_t<T>>;
    {t1 - t2} -> std::same_as<std::remove_cv_t<T>>;
    {t1 * t2} -> std::same_as<std::remove_cv_t<T>>;
    {t1 / t2} -> std::same_as<std::remove_cv_t<T>>;
};
```

### Common Types

#### `const_vector_t`

A `const_vector_t` is a template for generating `const std::vector<..>` types.

```cpp
template<typename T, typename Allocator = std::allocator<T>>
using const_vector_t = const std::vector<T, Allocator>;
```

It is not allowed for a `template template` parameter to be const qualified e.g. 
`template<template<typename> typename Container = const std::vector` will yield an error due to an invalid template parameter.
This is because `const` is used to qualify types while a template is used to generate types.
# MPEG

The goal is to implement MPEG style image compression using only the C++ standard library.

## Pipeline

1. File read
2. YCbCr Conversion
3. Chroma Subsampling
4. Tiled 2D DFT
5. Component Filtering
6. Huffman Coding
7. Output Formatting
8. File write

The initial plan is to implement stages 2 through 7 in C++ and then made accessible in Python via pybind11.

### 2. YCbCr Conversion

To take advantage of human vision being relatively less sensitive to color/chrominance changes relative to changes in brightness/luma, it is 
necessary to seperate out the luma component. How the 2 chroma channels are defined is an arbitrary decision that is worth experimentation.
The current defintions used by default in the code are taken from ITU-R BT.601 as taken from the 
[YCbCr Wiki page](https://en.wikipedia.org/wiki/YCbCr).

TODO: It is my understanding that chroma/chrominance can be used interchangeable but that it is best practice to make a distinction between
luma and luminance to avoid confusion with the concept of relative luminance. It is worth reading some color science to get a better
understanding.

## Type Definitions

### Matrix Concepts

#### `Can_Share`

The `Can_Share` concept is used to abstract the relationship between a `std::shared_ptr<...> Pointer` and the types `Underling`which can be used to generate other shared pointers that point at the same data as the provided pointer.

```cpp
template<typename Pointer, typename Underlying>
concept Can_Share = requires {std::same_as<std::remove_cvref_t<Pointer>, std::shared_ptr<Underlying>>;}
    || requires {std::same_as<std::remove_cvref_t<Pointer>, std::shared_ptr<const Underlying>>;};
```

Examples of such sharing are included below.

```cpp
std::shared_ptr<std::vector<int>> shared_vector (new std::vector<float>({1, 2, 3, 4, 5}));

std::shared_ptr<std::vector<int>> const_ptr = shared_vector; // valid!
std::shared_ptr<const std::vector<int>> const_ptr = shared_vector; // valid!
```

#### `Matrix_Like`

The `Matrix_Like` concept is used to abstract types which may be coerced into deep const matrices. A matrix must share the API of a 
`Multichannel_Matrix<...>` and have a type which satisfies the `Has_Arithmetic` concept.
```cpp
template<typename M, typename T>
concept Matrix_Like = requires(M m) {
    {m.data} -> Can_Share<T>;
    {m.shape()} -> std::same_as<Shape>;
    {m.order()} -> std::same_as<Order>;
    {m.strides()} -> std::same_as<Strides>;
    {m.size()} -> std::same_as<int>;
    Has_Arithmetic<T>;
};
```

### `Abstract_Matrix_Transformer`

An `Abstract_Matrix_Transformer` is a base class that transforms on matrices inherit from. An `Abstract_Matrix_Transformer` accepts
a given input shape and produces a given output shape. More detail on these types is included in the discussion of
`Multichannel_Matrix<...>`. The input is coerced to be deep const and then passed to an implementation of the transform that
accepts a deep const matrix.

```cpp
template<Has_Arithmetic T, Has_Arithmetic U, int Channels>
class Abstract_Matrix_Transformer {
private:
    virtual int transform_impl(const Multichannel_Matrix<T, Channels, const_vector_t>& in, Multichannel_Matrix<U, Channels> out) = 0;
public:
    const Shape input_shape;
    const Shape output_shape;
    
    const std::size_t input_size;
    const std::size_t output_size;

    ...

    template<Matrix_Like<T> M_T>
    int transform(M_T& in, Multichannel_Matrix<U, Channels> out){
        Multichannel_Matrix<T, Channels, const_vector_t> in_mat = Multichannel_Matrix<T, Channels, const_vector_t>::as_matrix(in);

        return transform_impl(in_mat, out);
    }
}
```

The deep const coercion works by creating a new matrix mirroring the `Matrix_Like` input, but with `const_vector_t` as the container type.
This allows the template function `transform<...>(...)` to accept both deep const input and non-const input while the implementation
only receives deep const input. The implementation is declared private as it is not a useful public interface.

A const reference is used to pass the value to `transform_impl(...)` so that the compiler will use the `const` version of any overloaded
methods. This also means that `transform_impl(...)` will not own a reference to the shared pointer which could introduce concern. The
implementation function relies on the wrapping function to continue owning a reference to the shared pointer so that the matrix will remain
valid. An issue could arise if the implementation stores a copy of the matrix reference without also storing the shared pointer seperatedly. This
is however disregarded as being bad code that is not worth consideration as of now.

### `Multichannel_Matrix`

```cpp
template<Has_Arithmetic T, int Channels, template<typename> typename Container = std::vector>
class Multichannel_Matrix {
private:
    std::shared_ptr<std::vector<T>> data;
    Shape shape_;
    Order order_;
    Strides strides_;
    int size_;
    ...
}
```


A multichannel matrix is a 3 dimensional matrix that is treated as a 2-D matrix with an additional "channel" dimension.
The channel dimension is intended to represent things like the color channel in an image i.e. for RGB each color is a seperate channel.

Public read-only variables follow the convention of appending an underscore to their name(`foo_`) for the underlying field and then only 
providing a getter which follows the same name (`foo()`). Fields are not declared `const` in order to make matrices trivially constructable and 
freely assignable.


**Current Work:**

1. Chroma subsampling will generate jagged multichannelmatrices so support for jagged matrices should be added. Partial support should be 
sufficient because the 2D submatrix for a given channel will not be jagged. Jaggedness will be introduced because the submatrix for each
channel might not be equal. Another approach would be to instead use 3 single channel matrices to avoid the definition of multichannel matrix
from getting too convoluted.

#### Template Parameters

1. `Has_Arithmetic T` is the type of each element stored in the underlying container.

2. `int Channel` the number of channels in the multichannel matrix.

3. `template<typename> typename Container = std::vector` the container in which to store the underlying data. This is taken as a
`template template` (a template passed to another template) which is later used to create the type `Container<T>`.

#### Data: `std::shared_ptr<std::vector<T>> data`
The underlying data is contained in a vector that is wrapped in a `std::shared_ptr` to provide support for matrices to point at the same data.
Data is accessed using the `T& index(...)` methods which provide a reference to the element pointed to by the arguments passed to the method.

#### Shape: `Shape shape_`

The shape of the 2D matrices that make up each channel of a `Multichannel_Matrix` is stored in a `struct Shape`. A simple struct is used to
package the fields together.
```cpp
struct Shape {
    int m; // m=height=# of rows
    int n; // n=width=# of cols
};
```

#### Order: `Order order_`

The internal arrangement of data within the underlying contiguous storage is specified by an `Order` struct. The fields of this struct hold
the order in which data is stored in memory. This can be seen as an extension of the row major vs column major distinction into 3 dimmensions.
For convenience, `static const` instances are provided for common arrangements so `Order::ROW_COL_CH` can be used instead of
`{Dimension::ROW, Dimension::Column, Dimension::Channel}`. `Order::ROW_COL_CH` corresponds to a contiguous series of row major matrices with
one row major matrix for each channel. First an entire row is stored. Once enough entire rows have been stored to complete the columns for the
included rows, then the process is repeated for the next channel.
```cpp
struct Order {
    Dimension first;
    Dimension second;
    Dimension third;
    
    static const Order ROW_COL_CH;
    static const Order CH_ROW_COL;
};

enum class Dimension {
    COLUMN,
    ROW,
    CHANNEL
};
```

Using $a_{i,j,k}$ to represent the row $i$, column $j$, and channel $k$. $2 \times 2 \times 2$ examples of how data gets stored are shown below.
Note that $i$ indexes rows which corresponds to a position within columns. Index $j$ indexes columns which corresponds to a position within
rows. The convention of starting matrix indices at 1 is only

| `data[]`            | `data[0]` | `data[1]` | `data[2]` | `data[3]` | `data[4]` | `data[5]` | `data[6]` | `data[7]` |
|---------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| `Order::ROW_CH_COL` | $a_{0,0,0}$ | $a_{0,1,0}$ | $a_{1,0,0}$ | $a_{1,1,0}$ | $a_{0,0,1}$ | $a_{0,1,1}$ | $a_{1,0,1}$ | $a_{1,1,1}$ |
| `Order::CH_ROW_COL` | $a_{0,0,0}$ | $a_{0,0,1}$ | $a_{0,1,0}$ | $a_{0,1,1}$ | $a_{1,0,0}$ | $a_{1,0,1}$ | $a_{1,1,0}$ | $a_{1,1,1}$ |

#### Strides: `Strides strides_`

A `Strides` struct stores the offset between consecutive elements in a given dimension of a multichannel matrix. This is calculated when a matrix
is created and is used for indexing i.e. when indexing `Multichannel_Matrix<T> mcm`, `mcm.index(i,j,k)` will return a reference to the element
at position `i*strides_.row + j*strides_.col + k*strides_.ch`.

```cpp
struct Strides {
    int row;
    int col;
    int ch;
};
```

#### Size: `int size_`

This field stores the size of the multichannel matrix which is calculated as `shape.m * shape.n * Channels`.


