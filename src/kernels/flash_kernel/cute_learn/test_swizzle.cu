#include<iostream>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>
#include<cute/swizzle.hpp>

using namespace cute;

int main(){
    const int BBits = 1; // mask length 1 bit
    const int MBase = 1; // row mask base 
    const int SShift = 2; // col mask shift left 2 bits
    auto s = Swizzle<BBits, MBase, SShift>{};
    auto l = make_layout(make_shape(4, 4), make_stride(4, 1));
    auto sol = composition(s, l);
    // auto l_o_s = composition(l, s); error
    print(sol);
    for(int i = 0;i < 16;i ++)printf("%d ", sol(i));puts("");
    // result : Sw<1,1,2> o _0 o (4,4):(4,1)0 4 10 14 1 5 11 15 2 6 8 12 3 7 9 13


    constexpr int kTileN = 128; // constexpr int kTileN = 128;
    constexpr int kTileM = 128; // constexpr int kTileN = 128;
    constexpr int kTileK = 32;
    constexpr int kStage = 5;
    constexpr int kShmLoadSwizzleM = 3;
    constexpr int kShmLoadSwizzleS = 3;
    constexpr int kShmLoadSwizzleB = 3;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    print(SmemLayoutAtom{});puts("");puts("");
    print(make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
    print(SmemLayoutA{});puts("");puts("");
    print(SmemLayoutB{});puts("");puts("");

    // Sw<3,3,3> o _0 o (_8,_32):(_32,_1)
    // (_128,_32,_5)Sw<3,3,3> o _0 o (_128,_32,_5):(_32,_1,_4096)
    // Sw<3,3,3> o _0 o (_128,_32,_5):(_32,_1,_4096)
    return 0;
}