#include<iostream>
#include<torch/torch.h>
#include<cute/stride.hpp>
#include<cute/layout.hpp>
#include<cute/tensor.hpp>
using namespace cute;

using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half>;
using G2SCopy = decltype(make_tiled_copy(
    g2s_copy_atom{},
    make_layout(make_shape(Int<32>{})),
    make_layout(make_shape(Int<8>{}))
    ));
