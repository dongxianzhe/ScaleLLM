#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

const int M = 32;
const int N = 16;
const int K = 16;
const int kTileM = 32;
const int kTileN = 16; 
const int kTileK = 16;
const int BLOCK_SIZE_M = kTileM;
const int BLOCK_SIZE_N = kTileN; 
const int BLOCK_SIZE_K = kTileK;

#define debug_tensor(t) { \
  if(thread0()){ \
    printf("%s:", #t);print(t);printf("\n"); \
  } \
}

// template<typename TiledMMA>
// __global__ void gemm_simple_kernel(half* Aptr, half* Bptr, half* Cptr){
//     using namespace cute;
//     Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
//     Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{}));
//     Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, Int<1>{}));
//     Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, _));
//     Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(0, _));
//     Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(0, 0));

//     TiledMMA tiledmma;
//     ThrMMA thr_mma = tiledmma.get_slice(0);
//     Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
//     Tensor tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
//     Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
//     auto tArA = thr_mma.partition_fragment_A(A(_, _, 0));  // (MMA, MMA_M, MMA_K)
//     auto tBrB = thr_mma.partition_fragment_B(B(_, _, 0));  // (MMA, MMA_K, MMA_N)
//     auto tCrC = thr_mma.partition_fragment_C(C(_, _));     // (MMA, MMA_M, MMA_N)
//     copy(tAgA(_, _, _), tArA);
//     copy(tBgB(_, _, _), tBrB);
//     gemm(tiledmma, tCrC, tArA, tBrB, tCrC);
//     copy(tCrC, tCgC);
// }

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void cute_mma_gemm_simple_kernel(T* Cptr, const T* Aptr, const T* Bptr, int m, int n, int k){
  using namespace cute;
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
  debug_tensor(A);
  debug_tensor(B);
  debug_tensor(C);
  int ix = blockIdx.x;
  int iy = blockIdx.y;
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  debug_tensor(gA);
  debug_tensor(gB);
  debug_tensor(gC);

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  Tensor tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_K, MMA_N)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
  clear(tCrC);
  debug_tensor(tAgA);
  debug_tensor(tBgB);
  debug_tensor(tCgC);
  debug_tensor(tArA);
  debug_tensor(tBrB);
  debug_tensor(tCrC);

  int num_tile_k = size<2>(gA);
  if(thread0())printf("num_tile_k = %d\n", num_tile_k);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; itile ++){
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC);
}

torch::Tensor gemm_simple(torch::Tensor A, torch::Tensor B){
    using namespace cute;
    using mma_op = SM80_16x8x16_F16F16F16F16_TN; // both A and B are k major
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMMA = decltype(
        make_tiled_mma(mma_atom{}, 
        make_layout(Shape<_2, _2, _1>{})
        // make_layout(Shape<_4, _1, _1>{})
    ));

    dim3 gridDim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M );
    dim3 blockDim(size(TiledMMA{}));
    torch::Tensor C = torch::zeros({M, N}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    using T = half;
    cute_mma_gemm_simple_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, TiledMMA><<<gridDim, blockDim>>>(
        static_cast<half*>(C.data_ptr()),
        static_cast<half*>(A.data_ptr()),
        static_cast<half*>(B.data_ptr()),
        M, N, K);
    // gemm_simple_kernel<TiledMMA><<<gridDim, blockDim>>>(
    //     static_cast<half*>(A.data_ptr()), 
    //     static_cast<half*>(B.data_ptr()), 
    //     static_cast<half*>(C.data_ptr())
    // );
    return C;
}

int main(){
    torch::Tensor A = torch::randn({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    torch::Tensor B = torch::randn({K, N}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    torch::Tensor C_ref = A.matmul(B);

    torch::Tensor C = gemm_simple(A, B.transpose(0, 1).contiguous());

    std::cout << C_ref.slice(0, 0, 16).slice(1, 0, 8) << std::endl;
    std::cout << C.slice(0, 0, 16).slice(1, 0, 8) << std::endl;

    if(torch::allclose(C, C_ref, 0.1, 0.1))puts("pass");
    else puts("failed");

}