#include<iostream>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>
using namespace cute;

// using MMA = decltype(make_tiled_mma(
//     SM80_16x8x16_F16F16F16F16_TN{},
//     make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})), 
//     Tile<_32, Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>, _16>{}));

using MMA = decltype(make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},
    make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})), 
    Tile<_32, _32, _16>{}));

using MMA = decltype(make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},
    make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})), 
    Tile<_32, _32, _16>{}));

int main(){
    print_latex(MMA{});
    return 0;
}