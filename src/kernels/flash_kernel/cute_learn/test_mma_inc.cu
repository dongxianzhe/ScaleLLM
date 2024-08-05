#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>
using namespace cute;

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;
constexpr int kStage = 3;
constexpr int kSmemLayoutCBatch = 2;
constexpr int kMmaPM = 32;
constexpr int kMmaPN = 32;
constexpr int kMmaPK = 16;

using SmemLayoutAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                make_stride(Int<kTileK>{}, Int<1>{}))));

// using SmemLayoutAtom = decltype(
//     make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
//                 make_stride(Int<kTileK>{}, Int<1>{})));

using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtom{},
                make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
using SmemLayoutB = decltype(
    tile_to_shape(SmemLayoutAtom{},
                make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));


using MMA = decltype(make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},
    make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})), 
    Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>{}));

using SmemLayoutAtomC = decltype(composition(
Swizzle<2, 3, 3>{},
make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), make_stride(Int<kMmaPN>{}, Int<1>{}))));
using SmemLayoutC = decltype(tile_to_shape(
    SmemLayoutAtomC{},
    make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

__global__ void kenrel(void *Aptr, const void *Bptr, int m, int n, int k) {
    int idx = threadIdx.x; int ix = blockIdx.x; int iy = blockIdx.y;
    extern __shared__ half shm_data[];
    half *Ashm = shm_data;
    half *Bshm = shm_data + cute::cosize(SmemLayoutA{});
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});  // (kTileM, kTileK, kStage) = (128, 32, 3)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutA{});  // (kTileM, kTileK, kStage) = (128, 32, 3)

    Tensor A = make_tensor(make_gmem_ptr((half *)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));  // (M, K) = (128, 32)
    Tensor B = make_tensor(make_gmem_ptr((half *)Bptr), make_shape(m, k), make_stride(k, Int<1>{}));  // (M, K) = (128, 32)

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, k) = (128, 32, 8)
    Tensor gB = local_tile(B, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, k) = (128, 32, 8)
    if(thread0()){
        printf("gA: ");print(gA);printf("\n");
        printf("gB: ");print(gB);printf("\n");
        printf("size gA: ");print(size(gA));printf("\n");
    }
    __syncthreads();

    auto g2s_tiled_copy_ab = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                    make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                    make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto g2s_thr_copy_ab = g2s_tiled_copy_ab.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_ab.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 32 / 32)
    auto tAsA_copy = g2s_thr_copy_ab.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    auto tAsB_copy = g2s_thr_copy_ab.partition_D(sB);  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    auto tAgB_copy = g2s_thr_copy_ab.partition_S(gB);  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 32 / 32)
    if(thread0()){
        printf("tAgA_copy: ");print(tAgA_copy);printf("\n");
        printf("tAsA_copy: ");print(tAsA_copy);printf("\n");
    }
    __syncthreads();

    // g2s
    copy(g2s_tiled_copy_ab, tAgA_copy(_, _, _, 0), tAsA_copy(_, _, _, 0));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for(int i = 0;i < 8;i ++){
        for(int j = 0;j < 4;j ++){
            tAsA_copy(i, j, 0, 0) = tAsA_copy(i, j, 0, 0) + __float2half(idx);
        }
    }

    __syncthreads();

    // s2r
    MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K) = (8, 4, 2) = (8, 128 / 32, 32 / 16)

    auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);

    auto tAsA = s2r_thr_copy_a.partition_S(sA);                   // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 2, 3)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);               // (CPY, CPY_M, CPY_K)         = (8, 4, 2)

    copy(s2r_tiled_copy_a, tAsA(_, _, _, 0), tCrA_view(_, _, _));
    if(thread0()){
        printf("thread0\n");
        print_tensor(tCrA_view);
    }
    __syncthreads();
    if(thread(32)){
        printf("thread32\n");
        print_tensor(tCrA_view);
    }
    __syncthreads();
    if(thread(64)){
        printf("thread64\n");
        print_tensor(tCrA_view);
    }

    // r2r

    // r2s

    // s2g




    // // s2s
    // for(int i = 0;i < 8;i ++){
    //     for(int j = 0;j < 4;j ++){
    //         tAsB_copy(i, j, 0, 0) = tAsA_copy(i, j, 0, 0) + __float2half(idx);
    //     }
    // }

    // // s2g
    // copy(tAsB_copy(_, _, _, 0), tAgB_copy(_, _, _, 0));
    // cp_async_fence();
    // cp_async_wait<0>();
    // __syncthreads();


    // // s2s one thread
    // if(thread0()){
    //     for(int i = 0;i < 8;i ++){
    //         for(int j = 0;j < 4;j ++){
    //             tAsB_copy(i, j, 0, 0) = tAsA_copy(i, j, 0, 0) + __float2half(1);
    //         }
    //     }
    // }
    // __syncthreads();


    // // s2g one thread
    // if(thread0()){ // s -> g
    //     for(int i = 0;i < 8;i ++){
    //         for(int j = 0;j < 4;j ++){
    //             tAgB_copy(i, j, 0, 0) = tAsB_copy(i, j, 0, 0);
    //         }
    //     }
    // }

    // // g2g one thread
    // if(thread0()){
    //     for(int i = 0;i < size(A);i ++)B.data()[i] = A.data()[i] + __float2half(1);
    // }
}

int main(){
    const int M = 128, N = 128, K = 32;
    auto A = torch::zeros({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto B = torch::zeros({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    puts("----------- before -------------------");
    std::cout << A << std::endl;
    puts("----------- kernel -------------------");
    dim3 block(size(MMA{}));
    dim3 grid(N / kTileN, M / kTileM);

    constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(half);
    kenrel<<<grid, block, 48 * 1024>>>(A.data_ptr(), B.data_ptr(), M, N, K);
    cudaDeviceSynchronize();
    puts("----------- after -------------------");
    std::cout << B << std::endl;
}