#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

namespace prefill_kernel_config{
    using namespace cute;

    using MMA = decltype(make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})), 
        Tile<Int<32>, Int<32>, Int<16>>{}));

    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));

    constexpr int kStage = 3;
    using SmemLayoutQ = decltype(
        tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<128>{}, Int<32>{}, Int<kStage>{})));
    using SmemLayoutK = decltype(
        tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<128>{}, Int<32>{}, Int<kStage>{})));
};

__global__ void prefill_kernel(half* Qptr, half* Kptr, half* Vptr, half* Sptr, int seq_len, int head_dim){
    using namespace cute;
    using namespace prefill_kernel_config;
    extern __shared__ half shm_data[];
    // 1. make q k v tensor
    Tensor Q = make_tensor(make_gmem_ptr(Qptr), make_shape(seq_len, head_dim), make_stride(seq_len, Int<1>{}));
    Tensor K = make_tensor(make_gmem_ptr(Kptr), make_shape(seq_len, head_dim), make_stride(seq_len, Int<1>{}));
    Tensor V = make_tensor(make_gmem_ptr(Vptr), make_shape(seq_len, head_dim), make_stride(seq_len, Int<1>{}));
    Tensor S = make_tensor(make_gmem_ptr(Sptr), make_shape(seq_len, seq_len ), make_stride(seq_len, Int<1>{}));

    // 2. divide q k
    Tensor gQ = local_tile(Q, make_tile(Int<128>{}, Int<32>{}), make_coord(blockIdx.x, _)); // (128, 32, 8)
    Tensor gK = local_tile(K, make_tile(Int<128>{}, Int<32>{}), make_coord(_, _));          // (128, 32, 8, 8) = (128, 32, 1024 / 128, 256 / 32)
    Tensor gS = local_tile(S, make_tile(Int<128>{}, Int<128>{}), make_coord(blockIdx.x, _));// (128, 128, 8) = (128, 128, 1024 / 128)

    // 3. Q K g2s
    Tensor sQ = make_tensor(make_smem_ptr(shm_data), SmemLayoutQ{}); 
    Tensor sK = make_tensor(make_smem_ptr(shm_data + cute::cosize(SmemLayoutK{})), SmemLayoutK{});

    auto g2s_tiled_copy_ab = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto g2s_thr_copy_ab = g2s_tiled_copy_ab.get_slice(threadIdx.x);
    auto tAgQ_copy = g2s_thr_copy_ab.partition_S(gQ);  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 256 / 32)
    auto tAsQ_copy = g2s_thr_copy_ab.partition_D(sQ);  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    auto tBgK_copy = g2s_thr_copy_ab.partition_S(gK);  // (CPY, CPY_N, CPY_K, n, k)      = (8, 4, 1, 8, 8) = (8, 128 / 32, 32 / 32, 1024 / 128, 256 / 32)
    auto tBsK_copy = g2s_thr_copy_ab.partition_D(sK);  // (CPY, CPY_N, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)

    // 4. mma
    MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrQ = thr_mma.partition_fragment_A(gQ(_, _, 0));     // (MMA, MMA_M, MMA_K) = (8, 2, 2) = (8, 128 / 64, 32 / 16)
    auto tCrK = thr_mma.partition_fragment_B(gK(_, _, 0, 0));  // (MMA, MMA_N, MMA_K) = (4, 16, 2) = (4, 128 / 8, 32 / 16)
    auto tCrS = thr_mma.partition_fragment_C(gS(_, _, 0));     // (MMA, MMA_M, MMA_N) = (4, 2, 16) = (4, 128 / 64, 128 / 8)

    // 5. Q K s2r
    auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    auto tAsQ = s2r_thr_copy_a.partition_S(sQ);                   // (CPY, CPY_M, CPY_K, kStage) = (8, 2, 2, 3)
    auto tCrQ_view = s2r_thr_copy_a.retile_D(tCrQ);               // (CPY, CPY_M, CPY_K)         = (8, 2, 2)

    auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    auto tBsK = s2r_thr_copy_b.partition_S(sK);                   // (CPY, CPY_M, CPY_K, kStage) = (16, 4, 2, 3)
    auto tCrK_view = s2r_thr_copy_b.retile_D(tCrK);               // (CPY, CPY_M, CPY_K)         = (16, 4, 2)

    // 6. pg2s 2 tile wait 1 tile, ps2r 1 tile
    int itile_to_read = 0;
    int ismem_read  = 0;  // s->r read  then increase
    int ismem_write = 0; // g->s write then increase
#pragma unroll
    for (int istage = 0; istage < kStage - 1 /*submit 2 tile*/; ++istage) { // 
        cute::copy(g2s_tiled_copy_ab, tAgQ_copy(_, _, _, istage), tAsQ_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_ab, tBgK_copy(_, _, _, 0, istage), tBsK_copy(_, _, _, istage)); // todo next block
        cp_async_fence();
        ++itile_to_read; ++ismem_write;
    }
    cp_async_wait<kStage - 2>(); // wait 1 submmited g->s done
    __syncthreads();

    int ik = 0; // prefetch first k tile
    cute::copy(s2r_tiled_copy_a, tAsQ(_, _, ik, ismem_read), tCrQ_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsK(_, _, ik, ismem_read), tCrK_view(_, _, ik));

    // 7. for each q tile and k tile 
    const int ntile = head_dim / 32;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrQ); // 32 / 16 = 2
    #pragma unroll
        for (int ik = 0; ik < nk; ++ik) { // each time compute one tile of the k block
            // 7.1 pg2s
            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }
            int ik_next = (ik + 1) % nk;
            cute::copy(s2r_tiled_copy_a, tAsQ(_, _, ik_next, ismem_read), tCrQ_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsK(_, _, ik_next, ismem_read), tCrK_view(_, _, ik_next));
            // 7.2 ps2r
            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_ab, tAgQ_copy(_, _, _, itile_to_read), tAsQ_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_ab, tBgK_copy(_, _, _, 0, itile_to_read), tBsK_copy(_, _, _, ismem_write));

                    ++ itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }
            // 7.3 compute
            cute::gemm(tiled_mma, tCrS, tCrQ(_, _, ik), tCrK(_, _, ik), tCrS);
        }
    }
}

torch::Tensor prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    // Q (seq_len, head_dim) head_dim major
    // K (seq_len, head_dim) head_dim major
    // V (seq_len, head_dim) head_dim major
    // 1.check todo
    const int seq_len = Q.size(0);
    const int head_dim = Q.size(1);
    // 2. each 4 warp cope with 128 seq length query
    dim3 griddim(seq_len / 128);
    dim3 blockdim(128);
    auto S = torch::zeros({seq_len, seq_len}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    prefill_kernel<<<griddim, blockdim, 49152>>>(
        static_cast<half*>(Q.data_ptr()),
        static_cast<half*>(K.data_ptr()),
        static_cast<half*>(V.data_ptr()),
        static_cast<half*>(S.data_ptr()),
        seq_len, head_dim
    );
    return S;
}

int main(){
    printf("hello world\n");
    return 0;
}