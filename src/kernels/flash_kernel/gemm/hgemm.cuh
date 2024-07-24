#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>
#include<cute/swizzle.hpp>

namespace gemm_pg2s_cute_config{
    using namespace cute;

    static constexpr int BLOCK_SIZE_M = 128;
    static constexpr int BLOCK_SIZE_N = 128;
    static constexpr int BLOCK_SIZE_K = 32;

    static constexpr int kStage = 3;

    static constexpr int kSmemLoadSwizzleB = 3;
    static constexpr int kSmemLoadSwizzleM = 3;
    static constexpr int kSmemLoadSwizzleS = 3;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSmemLoadSwizzleB, kSmemLoadSwizzleM, kSmemLoadSwizzleS>{},
        make_layout(make_shape(Int<8>{}, Int<BLOCK_SIZE_K>{}), make_stride(Int<BLOCK_SIZE_K>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BLOCK_SIZE_M>{}, Int<BLOCK_SIZE_K>{}, Int<kStage>{})
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<BLOCK_SIZE_N>{}, Int<BLOCK_SIZE_K>{}, Int<kStage>{})
    ));

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    using MyTiledMMA = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{}))
        ));
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half>;
    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
        ));
    using G2SCopyB = G2SCopyA;
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, half>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

}

__global__ void gemm_pg2s_cute_kernel(half* Cptr, half* Aptr, half* Bptr, int M, int N, int K){
    using namespace cute;
    using namespace gemm_pg2s_cute_config;
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, Int<1>{}));
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    Tensor gA = local_tile(A, make_tile(Int<BLOCK_SIZE_M>{}, Int<BLOCK_SIZE_K>{}), make_coord(iy, _)); // (BM, BK, NBK)
    Tensor gB = local_tile(B, make_tile(Int<BLOCK_SIZE_N>{}, Int<BLOCK_SIZE_K>{}), make_coord(ix, _)); // (BN, BK, NBK)
    Tensor gC = local_tile(C, make_tile(Int<BLOCK_SIZE_M>{}, Int<BLOCK_SIZE_N>{}), make_coord(iy, ix)); // (BM, BN)

    __shared__ half As[cute::cosize(SmemLayoutA{})];
    __shared__ half Bs[cute::cosize(SmemLayoutB{})];

    Tensor sA = make_tensor(make_smem_ptr(As), SmemLayoutA{}); // (BM, BK, kStage)
    Tensor sB = make_tensor(make_smem_ptr(Bs), SmemLayoutB{}); // (BN, BK, kStage)

    // MyTiledMMA tiled_mma;
    // ThrMMA thr_mma = tiled_mma.get_slice(idx);
    // auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)

    MyTiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(idx);
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC);           // (MMA, MMA_M, MMA_N)
    
    // fill zero for accumulator
    clear(tCrC);

    G2SCopyA g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    Tensor tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    Tensor tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)
    
    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy =
    g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)
}

torch::Tensor gemm_pg2s_cute(torch::Tensor A, torch::Tensor B){
    const int M = A.size(0);
    const int N = B.size(0);
    const int K = A.size(1);
    torch::Tensor C = torch::empty({M, N}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    using namespace gemm_pg2s_cute_config;

    dim3 gridDim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    dim3 blockDim(size(MyTiledMMA{}));
    gemm_pg2s_cute_kernel<<<gridDim, blockDim>>>(
        static_cast<half*>(C.data_ptr()),
        static_cast<half*>(A.data_ptr()),
        static_cast<half*>(B.data_ptr()),
        M, N, K);
    return C;
}