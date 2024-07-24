#include<iostream>
#include<cute/layout.hpp>

using namespace cute;

int main(){
    // A: shape is (9,32)
    auto layout_a = make_layout(make_shape (Int< 9>{}, make_shape (Int< 4>{}, Int<8>{})),
                                make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));
    // B: shape is (3,8)
    auto tiler = make_tile(Layout<_3,_3>{},           // Apply     3:3     to mode-0
                        Layout<Shape <_2,_4>,      // Apply (2,4):(1,8) to mode-1
                                Stride<_1,_8>>{});

    // ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))
    auto ld = logical_divide(layout_a, tiler); // ()
    // ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))
    auto zd = zipped_divide(layout_a, tiler);
    // tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...) shape ((3, 8), 3, 4)
    auto td = tiled_divide(layout_a, tiler);
    // flat_divide    : (TileM, TileN, RestM, RestN, L, ...) shape(3, 8, 3, 4)
    auto fd = flat_divide(layout_a, tiler);
    printf("logical divide: ");print(ld);puts("");
    printf("zipped_divide : ");print(zd);puts("");
    printf("tiled_divide  : ");print(td);puts("");
    printf("flat_divide   : ");print(fd);puts("");
    // logical divide: ((_3,_3),((_2,_4),(_2,_2))):((_177,_59),((_13,_2),(_26,_1)))
    // zipped_divide : ((_3,(_2,_4)),(_3,(_2,_2))):((_177,(_13,_2)),(_59,(_26,_1)))
    // tiled_divide  : ((_3,(_2,_4)),_3,(_2,_2)):((_177,(_13,_2)),_59,(_26,_1))
    // flat_divide   : (_3,(_2,_4),_3,(_2,_2)):(_177,(_13,_2),_59,(_26,_1))
}