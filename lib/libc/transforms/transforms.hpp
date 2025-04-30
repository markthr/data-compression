#ifndef __TRANSFORMS_H__
#define __TRANSFORMS_H__

#include <vector>
#include <concepts>

template<std::floating_point T, int n>
class Abstract_Transformer {
    public:
        /*
         *  Return 0 if successful, -1 otherwise
         */
        virtual int transform(const std::vector<T>& in, std::vector<T>& out) = 0;
};

template<std::floating_point T, int n>
class DCT_2 : Abstract_Transformer<T, n> {
    public:
        int transform(const std::vector<T>& in, std::vector<T>& out) override;
};

#include "dct_impl.hpp"

#endif