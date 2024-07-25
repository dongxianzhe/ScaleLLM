#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

using namespace cute;
// some equivalent method
// 
// using mma_op = SM80_16x8x16_F16F16F16F16_TN;
// using mma_traits = MMA_Traits<mma_op>;
// using mma_atom = MMA_Atom<mma_traits>;

// print(mma_atom{});puts("");
// print(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{});puts("");

// TiledMMA tiledmma1 = make_tiled_mma(mma_atom{}, make_layout(make_shape(2, 2, 1)));

// print(tiledmma1);

// TiledMMA tiledmma2 = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{}, make_layout(make_shape(2, 2, 1)));
// print(tiledmma2);

int main(){
    {
        puts("---------- SM70_8x8x4_F32F16F16F32_NT --------------------");
        TiledMMA tiledmma = make_tiled_mma(
            SM70_8x8x4_F32F16F16F32_NT{}, 
            Layout<Shape <_2,_2>, Stride<_2,_1>>{},
            Tile<_32,_32,_4>{});      // 32x32x4 tiler

        print(tiledmma);
    }
    {
        puts("---------- SM70_8x8x4_F32F16F16F32_NT permutated --------------------");
        TiledMMA tiledmma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
            Layout<Shape <_2,_2>, Stride<_2,_1>>{},       // 2x2 n-major layout of Atoms
            Tile<Layout<Shape <_4,_4,_2>, Stride<_1,_8,_4>>, // Permutation on M, size 32
            _32,                      // Permutation on N, size 32 identity
            _4>{});                   // Permutation on K, size 4 identity
        print(tiledmma);
    }
    {
        puts("---------- SM70_8x8x4_F32F16F16F32_NT --------------------");

        TiledMMA tiledmma = make_tiled_mma(
            SM80_16x8x16_F16F16F16F16_TN{}, 
            make_layout(make_shape(2, 2, 1))
        );

        print(tiledmma);
    }
}