#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

namespace tiled_gemm_128_128_32_config {
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
}

__global__ void tiled_gemm_kernel(void *Dptr, const void *Aptr, const void *Bptr, int m, int n, int k) {
  using namespace cute;using namespace tiled_gemm_128_128_32_config;
  int idx = threadIdx.x; int ix = blockIdx.x; int iy = blockIdx.y;

  Tensor A = make_tensor(make_gmem_ptr((half *)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));  // (M, K) = (1024, 256)
  Tensor B = make_tensor(make_gmem_ptr((half *)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));  // (N, K) = (1024, 256)
  Tensor D = make_tensor(make_gmem_ptr((half *)Dptr), make_shape(m, n), make_stride(n, Int<1>{}));  // (M, N) = (1024, 1024)

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, k) = (128, 32, 8)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));  // (kTileN, kTileK, k) = (128, 32, 8)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));  // (kTileM, kTileN) = (128, 128)

  extern __shared__ half shm_data[];
  half *Ashm = shm_data; half *Bshm = shm_data + cute::cosize(SmemLayoutA{});
  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});  // (kTileM, kTileK, kStage) = (128, 32, 3)
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});  // (kTileN, kTileK, kStage) = (128, 32, 3)

  MMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K) = (8, 4, 2) = (8, 128 / 32, 32 / 16)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K) = (4, 8, 2) = (4, 128 / 16, 32 / 16)
  auto tCrD = thr_mma.partition_fragment_C(gD);                 // (MMA, MMA_M, MMA_N) = (4, 4, 8) = (4, 128 / 32, 128 / 16)
  clear(tCrD);

  auto g2s_tiled_copy_ab = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{})));
  auto g2s_thr_copy_ab = g2s_tiled_copy_ab.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_ab.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 256 / 32)
  auto tAsA_copy = g2s_thr_copy_ab.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
  auto tBgB_copy = g2s_thr_copy_ab.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 256 / 32)
  auto tBsB_copy = g2s_thr_copy_ab.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)

  auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);                   // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 2, 3)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);               // (CPY, CPY_M, CPY_K)         = (8, 4, 2)

  auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);                   // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 2, 3)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);               // (CPY, CPY_M, CPY_K)         = (8, 4, 2)

  int itile_to_read = 0;
  int ismem_read = 0;  // s->r read  then increase
  int ismem_write = 0; // g->s write then increase
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) { // submit 2 tile
    cute::copy(g2s_tiled_copy_ab, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_ab, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read; ++ismem_write;
  }

  cp_async_wait<kStage - 2>(); // wait 1 submmited g->s done
  __syncthreads();

  int ik = 0; // prefetch first k tile
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  const int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) { // traverse k block
    int nk = size<2>(tCrA); // 32 / 16 = 2

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) { // each time compute one tile of the k block
      // prefetch next k tile
      int ik_next = (ik + 1) % nk;
      if (ik == nk - 1) { // if last k tile wait next submitted k block done
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage; // ready read pointer inc
      }
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));
      // prefetch next k block
      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_ab, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_ab, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));

          ++ itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }
      // caculate
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
  auto r2s_tiled_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N) = (8, 4, 4) // (8, 128 / 32, 128 / 32)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe) = (8, 1, 1, 2)

  auto s2g_tiled_copy_c = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, half>{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{})));
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe) = (8, 1, 1, 2)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N) = (8, 4, 4) = (8, 128 / 32, 128 / 32)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN) = (8, 16)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN) = (8, 16)

  int step = size<3>(tCsC_r2s); // 2
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) { // i = 0, 2, 4, 6, ... , 14
    // reg->smem
#pragma unroll
    for (int j = 0; j < step; ++j) { // 2 
      auto t = make_tensor_like<half>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);
      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();
    // smem -> global
#pragma unroll
    for (int j = 0; j < step; ++j) cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    __syncthreads();
  }
}

torch::Tensor tiled_gemm(torch::Tensor A, torch::Tensor B){
    using namespace tiled_gemm_128_128_32_config;
    const int M = A.size(0);const int N = B.size(0);const int K = A.size(1);
    torch::Tensor C = torch::empty({M, N}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    dim3 block(size(MMA{}));
    dim3 grid(N / kTileN, M / kTileM);

    constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(half);
    tiled_gemm_kernel<<<grid, block, kShmSize>>>(C.data_ptr(), A.data_ptr(), B.data_ptr(),M, N, K);
    return C;
}