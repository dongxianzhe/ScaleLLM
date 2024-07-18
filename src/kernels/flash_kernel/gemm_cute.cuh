#pragma once

#include<iostream>
#include"common.cuh"
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void cute_sgemm_naive_kernel(float* A, float* B, float* C, int M, int K, int N){
  using namespace cute;
  Tensor gA = make_tensor(A, make_shape(M, K), make_stride(K, 1)); // (M, K)
  Tensor gB = make_tensor(B, make_shape(K, N), make_stride(N, 1)); // (K, N)
  Tensor gC = make_tensor(C, make_shape(M, N), make_stride(N, 1)); // (M, N)
  Tensor block_A = local_tile(gA, make_shape(BLOCK_SIZE_M, K), blockIdx.x); // (BLOCK_SIZE_M, K)
  Tensor block_B = local_tile(gB, make_shape(K, BLOCK_SIZE_N), blockIdx.y); // (K, BLOCK_SIZE_N)
  Tensor block_C = local_tile(gC, make_shape(BLOCK_SIZE_M, BLOCK_SIZE_N), make_coord(blockIdx.x, blockIdx.y)); // (BLOCK_SIZE_M, BLOCK_SIZE_N)

  Tensor tile_A = local_tile(block_A, make_shape(1, K), threadIdx.x); // (1, K)
  Tensor tile_B = local_tile(block_B, make_shape(K, 1), threadIdx.y); // (K, 1)
  Tensor tile_C = local_tile(block_C, make_shape(1, 1), make_coord(threadIdx.x, threadIdx.y));
  float sum = 0;
  for(int k = 0;k < K;k ++){
    sum += tile_A(0, k) * tile_B(k, 0);
  }
  tile_C(0, 0) = sum;
}

torch::Tensor cute_sgemm_naive(torch::Tensor A, torch::Tensor B){
    const int BLOCK_SIZE_M = 8;
    const int BLOCK_SIZE_N = 8;

    CHECK_FATAL(A.dim() == 2, "A matrix dim should be 2");
    CHECK_FATAL(B.dim() == 2, "B matrix dim should be 2");
    CHECK_FATAL(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK_FATAL(A.scalar_type() == at::ScalarType::Float, "gemm naive only support float");
    CHECK_FATAL(B.scalar_type() == at::ScalarType::Float, "gemm naive only support float");
    CHECK_FATAL(A.scalar_type() == B.scalar_type(), "gemm matrix A and B should have same type");
    CHECK_FATAL(A.size(0) % BLOCK_SIZE_M == 0, "M should mod BLOCK_SIZE_M equal zero");
    CHECK_FATAL(B.size(1) % BLOCK_SIZE_N == 0, "N should mod BLOCK_SIZE_N equal zero");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());
    dim3 gridDim(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
    dim3 blockDim(BLOCK_SIZE_M, BLOCK_SIZE_N);

    cute_sgemm_naive_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N><<<gridDim, blockDim>>>(
      static_cast<float*>(A.data_ptr()),
      static_cast<float*>(B.data_ptr()),
      static_cast<float*>(C.data_ptr()),
      M, K, N);
    return C;
}

#define debug_tensor(t) { \
  if(thread0()){ \
    printf("%s:", #t);print(t);printf("\n"); \
  } \
}

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


torch::Tensor cute_mma_gemm_simple(torch::Tensor A, torch::Tensor B){
  int M = A.size(0), N = B.size(1), K = A.size(1);
  torch::Tensor B_Kmajor = B.transpose(0, 1);
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

  torch::Tensor C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));

  dim3 gridDim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
  dim3 blockDim(size(TiledMMA{}));
  std::cout << "gridDim:" << gridDim << std::endl;
  std::cout << "blockDim:" << blockDim << std::endl;

  using T = half;
  cute_mma_gemm_simple_kernel<T, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, TiledMMA><<<gridDim, blockDim>>>(
    static_cast<half*>(C.data_ptr()),
    static_cast<half*>(A.data_ptr()),
    static_cast<half*>(B_Kmajor.data_ptr()),
    M, N, K);
  return C;
}