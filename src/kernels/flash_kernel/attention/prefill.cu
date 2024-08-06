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

    // 2. divide q k
    Tensor gQ = local_tile(Q, make_tile(Int<128>{}, Int<32>{}), make_coord(blockIdx.x, _)); // (128, 32, 8)
    Tensor gK = local_tile(K, make_tile(Int<128>{}, Int<32>{}), make_coord(_, _));          // (128, 32, 8, 8) = (128, 32, 1024 / 128, 256 / 32)

    // 3. Q K g2s
    Tensor sQ = make_tensor(make_smem_ptr(shm_data), SmemLayoutQ{}); 
    Tensor sK = make_tensor(make_smem_ptr(shm_data + cute::cosize(SmemLayoutQ{})), SmemLayoutQ{});

    auto g2s_tiled_copy_ab = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto g2s_thr_copy_ab = g2s_tiled_copy_ab.get_slice(threadIdx.x);
    auto tAgQ_copy = g2s_thr_copy_ab.partition_S(gQ);  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 256 / 32)
    auto tAsQ_copy = g2s_thr_copy_ab.partition_D(sQ);  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    auto tBgK_copy = g2s_thr_copy_ab.partition_S(gK);  // (CPY, CPY_N, CPY_K, k)      = (8, 4, 1, 8, 8) = (8, 128 / 32, 32 / 32, 1024 / 128, 256 / 32)
    auto tBsK_copy = g2s_thr_copy_ab.partition_D(sK);  // (CPY, CPY_N, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)

    // 4. Q K s2r
    MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrQ = thr_mma.partition_fragment_A(gQ(_, _, 0));     // (MMA, MMA_M, MMA_K) = (8, 2, 2) = (8, 128 / 64, 32 / 16)
    auto tCrK = thr_mma.partition_fragment_B(gK(_, _, 0, 0));  // (MMA, MMA_N, MMA_K) = (4, 16, 2) = (4, 128 / 8, 32 / 16)
    // Tensor c = gK(_, _, 0, 0);

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