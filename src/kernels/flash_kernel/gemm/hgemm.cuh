#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>
#include<cute/swizzle.hpp>

namespace gemm_asynccp_ldmatrix_tensorcore_cute_config{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int kStage = 3;
}

__global__ void gemm_asynccp_ldmatrix_tensorcore_cute_kernel(half* Aptr, half* Bptr, half* Cptr, int M, int N, int K){
    using namespace cute;
    using namespace gemm_asynccp_ldmatrix_tensorcore_cute_config;
    // 1. A, B, C
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_layout(make_shape(N, K), make_stride(K, Int<1>{})));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_layout(make_shape(M, N), make_stride(N, Int<1>{})));

    // 2. thread block gA, gB
    // 2.1 A (1024, 256) -- zipdivide (128, 32) -> ((128, 32), (8, 8)) -- select((_, _), (blockIdx.y, _)) -> (128, 32, 8)
    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.y, _));
    // 2.2 B (1024, 256) -- zipdivide (128, 32) -> ((128, 32), (8, 8)) -- select((_, _), (blockIdx.x, _)) -> (128, 32, 8)
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(blockIdx.x, _));
    // 2.3 C (1024, 1024) -- zipdivide (128, 128) -> ((128, 128), (8, 8)) -- select((_, _), (blockIdx.y, blockIdx.x))
    Tensor gC = local_tile(C, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.y, blockIdx.x));

    // compute
    TiledMMA tiled_mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{}, // A is k major, B is k major
        make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})), // 4 warps 
        Tile<Int<32>, Int<32>, Int<16>>{}
    );
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (8, 4, 2)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (4, 8, 2)
    Tensor tCrC = thr_mma.partition_fragment_C(gC(_, _));  // (4, 4, 8)

    // 3. copy from global to smem tAgA -> tAsA, tBgB -> tBsB
    // 3.1 Copy_Atom
    //   ThrID:        _1:_0
    //   ValLayoutSrc: (_1,_8):(_0,_1) // 一条指令负责8个half
    //   ValLayoutDst: (_1,_8):(_0,_1)
    //   ValLayoutRef: (_1,_8):(_0,_1)
    //   ValueType:    16b
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half>;
    // 3.2 tiled copy
    // Tiler_MN:       (_32,_32) // 一个tile是32 * 32个half，共4个tile
    // TiledLayout_TV: ((_4,_32),_8):((_256,_1),_32) T0和T1相差8列,32行，每个线程负责K轴的8个连续的half
    using g2s_copy_a = decltype(make_tiled_copy(g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
        ));
    using g2s_copy_b = g2s_copy_a;
    // 3.3 shared memory
    extern __shared__ half smem_data[];
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(
            make_shape(Int<8>{}, Int<BK>{}),
            make_stride(Int<BK>{}, Int<1>{})
            )));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{}, 
        make_shape(Int<BM>{}, Int<BK>{}, Int<kStage>{})
        ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{}, 
        make_shape(Int<BN>{}, Int<BK>{}, Int<kStage>{})
        ));
    half* Asmem = smem_data;
    half* Bsmem = smem_data + cute::cosize(SmemLayoutA{});
    Tensor sA = make_tensor(
        make_smem_ptr(Asmem), SmemLayoutA{});
    Tensor sB = make_tensor(
        make_smem_ptr(Asmem), SmemLayoutB{});
    // 3.4 partition
    g2s_copy_a g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
    Tensor tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (8, 4, 1, 8)
    Tensor tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (8, 4, 1, 3)

    g2s_copy_b g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(threadIdx.x);
    Tensor tBgB_copy = g2s_thr_copy_a.partition_S(gB); // (8, 4, 1, 8)
    Tensor tBsB_copy = g2s_thr_copy_a.partition_D(sB); // (8, 4, 1, 3)
    // 4. copy from smem to reg tAsA->tArA,tBsB->tBrB
    // s2r_tiled_copy_a:TiledCopy
    //   Tiler_MN:       (_32,_16)  // 虽然又4个warp，但是从smem到reg拷贝的时候有些warp拷贝的内容是相同的。每个warp拷贝的四个小矩阵32x8
    //   TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
    // Copy_Atom
    //   ThrID:        _32:_1
    //   ValLayoutSrc: (_32,_8):(_8,_1) // 32个线程，一个线程load 8个half,4个8x8的小矩阵
    //   ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64)) // 32个线程，每个每个寄存器存两个数，4个寄存器，其中每个寄存器内偏移为shared memory物理地址的1，寄存器间偏移为shared memory的物理地址的64
    //   ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
    //   ValueType:    16b
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, half>;
    using s2r_copy_atom_a = s2r_copy_atom;
    using s2r_copy_atom_b = s2r_copy_atom;
    TiledCopy s2r_tiled_copy_a = make_tiled_copy_A(s2r_copy_atom_a{}, tiled_mma);
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    Tensor tAsA = s2r_thr_copy_a.partition_S(sA); // (8, 4, 2, 3)
    Tensor tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (8, 4, 2)

    TiledCopy s2r_tiled_copy_b = make_tiled_copy_B(s2r_copy_atom_b{}, tiled_mma);
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    Tensor tBsB = s2r_thr_copy_a.partition_S(sA); // (8, 4, 2, 3)
    Tensor tCrB_view = s2r_thr_copy_a.retile_D(tCrA); // (8, 4, 2)

    // 5. compute gemm tCrA, tCrB -> tCrC
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;
    for(int istage = 0; istage < kStage - 1; ++ istage){
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();
        ++ itile_to_read;
        ++ ismem_write;
    }
    cp_async_wait<kStage - 2>();
    __syncthreads();
    int ik = 0;
    copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));
    int ntile = K / BK;
}

torch::Tensor gemm_asynccp_ldmatrix_tensorcore_cute(torch::Tensor A, torch::Tensor B){
    using namespace gemm_asynccp_ldmatrix_tensorcore_cute_config;
    const int M = A.size(0);
    const int N = B.size(0);
    const int K = A.size(1);
    torch::Tensor C = torch::empty({M, N}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    dim3 gridDim(N / BN, M / BM);
    // dim3 blockDim();
    return C;
}