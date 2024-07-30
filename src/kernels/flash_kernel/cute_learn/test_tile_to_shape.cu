#include<iostream>
#include<cute/tensor.hpp>
#include<cute/layout.hpp>

int main(){
    using namespace cute;
    Layout l = make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    Layout s = tile_to_shape(l, make_shape(Int<128>{}, Int<64>{}, Int<3>{}));
    printf("l :");print(l);printf("\n");
    printf("s :");print(s);printf("\n");
    return 0;
}