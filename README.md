# Data Compression

The goal of this project is to implement data compression algorithms using only the C++ standard library.
These compression algorithms and many of the transforms necessary to create them will be made available as a Python library using pybind11.

Currently, focus is on MPEG style image compression. MP3 style audio compression is an eventual goal.

## Base Transformations

### FFT: Fast Fourier Transform

### DFT-II: Discrete Fourier Transform

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

### Multichannel_Matrix

```
template<typename T, int Channels, std::size_t Extent>
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
channel might not be equal.

2. A method for functions which accept a multichannel matrix to coerce deep const. It is desirable for a function to specify that it will
not modify any data in a matrix it takes as a parameter, but `Multichannel_Matrix<T>` cannot be coerced to `Multichannel_Matrix<const T>`.
This issue is discussed [in this blog](https://brevzin.github.io/c++/2021/09/10/deep-const/).

3. Add a shallow copy constructor to leverage the ability to use `shared_ptr` to make non-owning matrices.


#### Data: `std::shared_ptr<std::vector<T>> data`
The underlying data is contained in a vector that is wrapped in a `std::shared_ptr` to provide support for matrices to point at the same data.
Data is accessed using the `T& index(...)` methods which provide a reference to the element pointed to by the arguments passed to the method.

#### Shape: `Shape shape_`

The shape of the 2D matrices that make up each channel of a `Multichannel_Matrix` is stored in a `struct Shape`. A simple struct is used to
package the fields together.
```
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
```
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

```
struct Strides {
    int row;
    int col;
    int ch;
};
```

#### Size: `int size_`

This field stores the size of the multichannel matrix which is calculated as `shape.m * shape.n * Channels`.


