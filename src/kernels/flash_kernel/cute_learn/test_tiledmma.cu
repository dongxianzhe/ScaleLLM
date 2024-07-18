#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

#define debug_tensor(t) {printf("%s:", #t);print(t);printf("\n");}

int main(){
    const int M = 1024;
    const int K = 1024;
    const int m = M;
    const int k = K;
    torch::Tensor A_elements = torch::arange(M * K);
    torch::Tensor A_Matrix = A_elements.reshape({M, K}).to(torch::kHalf);
    
    puts("------------libtorch tensor index access------------------");
    for(int i = 0;i < 4;i ++){
        for(int j = 0;j < 4;j ++){
            auto element = A_Matrix.index({i, j});
            printf("%f ", element.item<float>());
        }puts("");
    }

    puts("---------- pointer access--------------------");
    half* Aptr = static_cast<half*>(A_Matrix.data_ptr());
    for(int i = 0;i < 9;i ++){
        for(int j = 0;j < 4;j ++){
            printf("%f ", static_cast<float>(Aptr[i * K + j]));
        }puts("");
    }
    puts("-----------tiled mma-------------------");
    using namespace cute;
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 256;
    const int BLOCK_SIZE_K = 32;
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMMA = decltype(
        make_tiled_mma(mma_atom{},
        make_layout(Shape<_2, _2, _1>{})
        // make_layout(Shape<_4, _1, _1>{})
        ));
    printf("TiledMMA:"); print(TiledMMA{}); printf("\n");
    
    puts("----------tile --------------------");
    const int kTileM = 128;
    const int kTileK = 32;
    Tensor A = make_tensor(Aptr, make_shape(m, k), make_stride(k, Int<1>{}));
    debug_tensor(A);
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, _));
    debug_tensor(gA);

    for(int i = 0;i < 4;i ++){
        for(int j = 0;j < 4;j ++){
            printf("%f ",__half2float(A[make_coord(i, j)]));
        }puts("");
    }
    for(int i = 0;i < 4;i ++){
        for(int j =0;j < 4;j ++){
            printf("%f ", __half2float(gA[make_coord(i, j, 0)]));
        }puts("");
    }
    for(int i = 0;i < 4;i ++){
        for(int j =0;j < 4;j ++){
            printf("%f ", __half2float(gA[make_coord(i, j, 1)]));
        }puts("");
    }

    puts("---------- fragment --------------------");
    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(0);
    Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    debug_tensor(tAgA);
    debug_tensor(tArA);

    for(int i = 0;i < 8;i ++){
        printf("%f ", __half2float(tAgA[make_coord(i, 0, 0, 0)]));
    }puts("");
    tArA[make_coord(0, 0, 0)] = 1;
    for(int i = 0;i < 8;i ++){
        printf("%f ",tArA[make_coord(i, 0, 0)]);
    }puts("");
}