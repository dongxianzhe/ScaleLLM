#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

void test(torch::Tensor a, torch::Tensor b, std::string name){
    if (a.is_cuda())a = a.to(torch::kCPU);
    if (b.is_cuda())b = b.to(torch::kCPU);
    float eps = 1e-1;
    if (a.allclose(b, eps, eps)) {
        std::cout << name << ": pass" << std::endl;
    } else {
        std::cout << name << ": fail" << std::endl;
    }
}

template<
    typename g2s_tiled_copy_a_type, 
    typename gA_g2s_type, 
    typename sA_g2s_type, 
    typename s2r_tiled_copy_a_type, 
    typename sA_s2r_type, 
    typename rA_s2r_type, 
    typename g2s_tiled_copy_b_type, 
    typename gB_g2s_type, 
    typename sB_g2s_type, 
    typename s2r_tiled_copy_b_type, 
    typename sB_s2r_type, 
    typename rB_s2r_type, 
    typename tiled_mma_type,
    typename rA_type, 
    typename rB_type, 
    typename rC_type
>
__device__ void gemm_M128N128_pipeline(
    g2s_tiled_copy_a_type& g2s_tiled_copy_a, 
    gA_g2s_type&           gA_g2s          ,  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 256 / 32)
    sA_g2s_type&           sA_g2s          ,  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    s2r_tiled_copy_a_type& s2r_tiled_copy_a, 
    sA_s2r_type&           sA_s2r          ,  // (CPY, CPY_M, CPY_K, kStage) = (8, 2, 2, 3)
    rA_s2r_type&           rA_s2r          ,  // (CPY, CPY_M, CPY_K)         = (8, 2, 2)
    g2s_tiled_copy_b_type& g2s_tiled_copy_b,
    gB_g2s_type&           gB_g2s          ,  // (CPY, CPY_N, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 1024 / 128, 256 / 32)
    sB_g2s_type&           sB_g2s          ,  // (CPY, CPY_N, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    s2r_tiled_copy_b_type& s2r_tiled_copy_b, 
    sB_s2r_type&           sB_s2r          ,  // (CPY, CPY_M, CPY_K, kStage) = (16, 4, 2, 3)
    rB_s2r_type&           rB_s2r          ,  // (CPY, CPY_M, CPY_K)         = (16, 4, 2)
    tiled_mma_type&        tiled_mma       ,
    rA_type&               rA              ,  // (MMA, MMA_M, MMA_K)         = (8, 2, 2)  = (8, 128 / 64, 32 / 16)
    rB_type&               rB              ,  // (MMA, MMA_N, MMA_K)         = (4, 16, 2) = (4, 128 / 8, 32 / 16)
    rC_type&               rC                 // (MMA, MMA_M, MMA_N)         = (4, 2, 16) = (4, 128 / 64, 128 / 8)
    ){
    using namespace cute;
    constexpr int kStage = 3;

    int itile_to_read = 0; 
    int ismem_read  = 0;  // s2r read then increase
    int ismem_write = 0; // g2s write then increase
#pragma unroll
    for (int istage = 0; istage < kStage - 1 /*submit 2 tile*/; ++istage) { // 
        cute::copy(g2s_tiled_copy_a, gA_g2s(_, _, _, istage), sA_g2s(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, gB_g2s(_, _, _, istage), sB_g2s(_, _, _, istage)); // todo next block
        cp_async_fence();
        ++itile_to_read; ++ismem_write;
    }
    cp_async_wait<kStage - 2>(); // wait 1 submmited g->s done
    __syncthreads();

    int ik = 0; // prefetch first k tile
    cute::copy(s2r_tiled_copy_a, sA_s2r(_, _, ik, ismem_read), rA_s2r(_, _, ik));
    cute::copy(s2r_tiled_copy_b, sB_s2r(_, _, ik, ismem_read), rB_s2r(_, _, ik));

    // 7. for each q tile and k tile 
    const int ntile = size<3>(gA_g2s);
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(rA); // 32 / 16 = 2
    #pragma unroll
        for (int ik = 0; ik < nk; ++ik) { // each time compute one tile of the k block
            // 7.1 pg2s
            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }
            int ik_next = (ik + 1) % nk;
            cute::copy(s2r_tiled_copy_a, sA_s2r(_, _, ik_next, ismem_read), rA_s2r(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, sB_s2r(_, _, ik_next, ismem_read), rB_s2r(_, _, ik_next));
            // 7.2 ps2r
            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, gA_g2s(_, _, _, itile_to_read), sA_g2s(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, gB_g2s(_, _, _, itile_to_read), sB_g2s(_, _, _, ismem_write));

                    ++ itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                cp_async_fence();
            }
            // 7.3 compute
            cute::gemm(tiled_mma, rC, rA(_, _, ik), rB(_, _, ik), rC);
        }
    }
}

__global__ void prefill_kernel(half* Qptr, half* Kptr, half* Vptr, half* Sptr, half* Optr, int seq_len, int head_dim){
    using namespace cute;
    extern __shared__ half shm_data[];
    // 1. make q k v s o tensor
    Tensor Q = make_tensor(make_gmem_ptr(Qptr), make_shape(seq_len, head_dim), make_stride(head_dim, Int<1>{}));
    Tensor K = make_tensor(make_gmem_ptr(Kptr), make_shape(seq_len, head_dim), make_stride(head_dim, Int<1>{}));
    Tensor S = make_tensor(make_gmem_ptr(Sptr), make_shape(seq_len, seq_len ), make_stride(seq_len, Int<1>{}));

    Tensor V = make_tensor(make_gmem_ptr(Vptr), make_shape(head_dim, seq_len), make_stride(Int<1>{}, head_dim));
    Tensor O = make_tensor(make_gmem_ptr(Optr), make_shape(seq_len, head_dim), make_stride(head_dim, Int<1>{}));

    // 2. divide q k
    Tensor gQ = local_tile(Q, make_tile(Int<128>{}, Int<32>{}), make_coord(blockIdx.x, _)); // (128, 32, 8)
    Tensor gK = local_tile(K, make_tile(Int<128>{}, Int<32>{}), make_coord(_, _));          // (128, 32, 8, 8) = (128, 32, 1024 / 128, 256 / 32)
    Tensor gS = local_tile(S, make_tile(Int<128>{}, Int<128>{}), make_coord(blockIdx.x, _));// (128, 128, 8) = (128, 128, 1024 / 128)
    Tensor gO = local_tile(O, make_tile(Int<128>{}, Int<128>{}), make_coord(blockIdx.x, _));// (128, 128, 2) = (128, 128, 256 / 2)

    // if(thread0()){
    //     printf("gQ: ");print(gQ);printf("\n"); // (_128,_32,8):(256,_1,_32)
    //     printf("gK: ");print(gK);printf("\n"); // (_128,_32,8,8):(256,_1,32768,_32)
    //     printf("gS: ");print(gS);printf("\n"); // (_128,_128,8):(1024,_1,_128)
    // }

    // 3. Q K g2s
    constexpr int kStage = 3;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));
    using SmemLayoutQ = decltype(
        tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<128>{}, Int<32>{}, Int<kStage>{})));
    using SmemLayoutK = decltype(
        tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<128>{}, Int<32>{}, Int<kStage>{})));
    Tensor sQ = make_tensor(make_smem_ptr(shm_data), SmemLayoutQ{}); 
    Tensor sK = make_tensor(make_smem_ptr(shm_data + cute::cosize(SmemLayoutQ{})), SmemLayoutK{});

    auto g2s_tiled_copy_ab = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto g2s_thr_copy_ab = g2s_tiled_copy_ab.get_slice(threadIdx.x);
    auto gQ_g2s = g2s_thr_copy_ab.partition_S(gQ);  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 256 / 32)
    auto sQ_g2s = g2s_thr_copy_ab.partition_D(sQ);  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    auto gK_g2s = g2s_thr_copy_ab.partition_S(gK);  // (CPY, CPY_N, CPY_K, n, k)      = (8, 4, 1, 8, 8) = (8, 128 / 32, 32 / 32, 1024 / 128, 256 / 32)
    auto sK_g2s = g2s_thr_copy_ab.partition_D(sK);  // (CPY, CPY_N, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    // if(thread0()){
    //     printf("sQ: ");print(sQ);printf("\n"); // Sw<3,3,3> o _0 o (_128,_32,_3):(_32,_1,_4096)
    //     printf("sK: ");print(sK);printf("\n"); // Sw<3,3,3> o _0 o (_128,_32,_3):(_32,_1,_4096)
    //     printf("gQ_g2s: ");print(gQ_g2s);printf("\n"); // ((_8,_1),_4,_1,8):((_1,_0),8192,_0,_32)
    //     printf("sQ_g2s: ");print(sQ_g2s);printf("\n"); // ((_8,_1),_4,_1,_3):((_1,_0),_1024,_0,_4096)
    //     printf("gK_g2s: ");print(gK_g2s);printf("\n"); // ((_8,_1),_4,_1,8,8):((_1,_0),8192,_0,32768,_32)
    //     printf("sK_g2s: ");print(sK_g2s);printf("\n"); // ((_8,_1),_4,_1,_3):((_1,_0),_1024,_0,_4096)
    // }

    for(int sid = 0; sid < 1024 / 128; sid ++){ // each time compute one S block
        // 4. mma
        using MMA = decltype(make_tiled_mma(
            SM80_16x8x16_F16F16F16F16_TN{},
            make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})), 
            Tile<Int<64>, Int<32>, Int<16>>{}));
        MMA tiled_mma;
        auto thr_mma = tiled_mma.get_slice(threadIdx.x);
        auto rQ = thr_mma.partition_fragment_A(gQ(_, _, 0));     // (MMA, MMA_M, MMA_K) = (8, 2, 2) = (8, 128 / 64, 32 / 16)
        auto rK = thr_mma.partition_fragment_B(gK(_, _, sid, 0));// (MMA, MMA_N, MMA_K) = (4, 16, 2) = (4, 128 / 8, 32 / 16)
        auto rS = thr_mma.partition_fragment_C(gS(_, _, 0));     // (MMA, MMA_M, MMA_N) = (4, 2, 16) = (4, 128 / 64, 128 / 8)

        // 5. Q K s2r
        auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
        auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
        auto sQ_s2r = s2r_thr_copy_a.partition_S(sQ);            // (CPY, CPY_M, CPY_K, kStage) = (8, 2, 2, 3)
        auto rQ_s2r = s2r_thr_copy_a.retile_D(rQ);               // (CPY, CPY_M, CPY_K)         = (8, 2, 2)

        auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
        auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
        auto sK_s2r = s2r_thr_copy_b.partition_S(sK);            // (CPY, CPY_M, CPY_K, kStage) = (16, 4, 2, 3)
        auto rK_s2r = s2r_thr_copy_b.retile_D(rK);               // (CPY, CPY_M, CPY_K)         = (16, 4, 2)

        auto gK_g2s_tile = gK_g2s(_, _, _, sid, _);
        // 6. pg2s 2 tile wait 1 tile, ps2r 1 tile
        gemm_M128N128_pipeline(
            g2s_tiled_copy_ab, 
            gQ_g2s          ,
            sQ_g2s          ,
            s2r_tiled_copy_a,
            sQ_s2r          ,
            rQ_s2r          ,
            g2s_tiled_copy_ab,
            gK_g2s_tile     ,
            sK_g2s          ,
            s2r_tiled_copy_b,
            sK_s2r          ,
            rK_s2r          ,
            tiled_mma       ,
            rQ              ,
            rK              ,
            rS               
        );

        // [ut] copy S back to test S
        auto sS = make_tensor(make_smem_ptr(shm_data), make_layout(make_shape(Int<64>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})));
        auto r2s_tiled_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma); // (64, 32)
        auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x); // CPY = 16
        auto rS_r2s = r2s_thr_copy_c.retile_S(rS);   // (CPY, CPY_M, CPY_N) = (16, 2, 4) // (16, 128 / 64, 128 / 32)
        auto sS_r2s = r2s_thr_copy_c.partition_D(sS);  // (CPY, _1, _1) = (16, 1, 1)

        auto s2g_tiled_copy_c = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, half>{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
        auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(threadIdx.x);
        auto sS_s2g = s2g_thr_copy_c.partition_S(sS);  // (CPY, CPY_M, CPY_N) = (8, 2, 1)
        auto gS_s2g = s2g_thr_copy_c.partition_D(gS);  // (CPY, CPY_M, CPY_N) = (8, 4, 4, 8) = (8, 128 / 32, 128 / 32, 1024 / 128)
        for(int i = 0;i < 2;i ++){
            for(int j = 0;j < 4;j ++){
                copy(r2s_tiled_copy_c, rS_r2s(_, i, j), sS_r2s(_, 0, 0));
                __syncthreads();
                copy(s2g_tiled_copy_c, sS_s2g(_, 0, j), gS_s2g(_, i * 2, j, sid));
                copy(s2g_tiled_copy_c, sS_s2g(_, 1, j), gS_s2g(_, i * 2 + 1, j, sid));
                __syncthreads();
            }
        }
        cp_async_wait<0>();
        __syncthreads();

        // // 1. make V O tensor
        // Tensor gVsid = local_tile(V, make_tile(Int<256>{}, Int<128>{}), make_coord(0, sid)); // (256, 128)
        // Tensor gV = local_tile(gVsid, make_tile(Int<128>{}, Int<32>{}), make_coord(_, _));   // (128, 32, 2, 4)
        // // 2. g2s V O partition
        // using SmemLayoutV = decltype(make_layout(make_shape(Int<128>{}, Int<32>{}, Int<kStage>{}), make_stride(Int<1>{}, Int<32>{}, Int<4096>{})));
        // Tensor sV = make_tensor(make_smem_ptr(shm_data + cute::cosize(SmemLayoutQ{})), SmemLayoutV{}); // (128, 32, 3)
        // auto g2s_tiled_copy_V = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
        //                             make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<1>{}, Int<32>{})),
        //                             make_layout(make_shape(Int<8>{}, Int<1>{})));
        // auto g2s_thr_copy_V = g2s_tiled_copy_V.get_slice(threadIdx.x);
        // auto gV_g2s = g2s_thr_copy_V.partition_S(gV);  // (8, 1, 4, 2, 4)
        // auto sV_g2s = g2s_thr_copy_V.partition_D(sV);  // (8, 1, 4, 3)

        // auto rV = thr_mma.partition_fragment_B(gV(_, _, 0, 0));   // (4, 16, 2)
        // auto rO = thr_mma.partition_fragment_C(gO(_, _, _));      // (4, 2, 16, 2)

        // // 2. s2r V O partition
        // auto s2r_tiled_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, half>{}, tiled_mma);
        // auto s2r_thr_copy_V = s2r_tiled_copy_V.get_slice(threadIdx.x);
        // auto sV_s2r = s2r_thr_copy_V.partition_S(sV);            // (16, 4, 2, 3)
        // auto rV_s2r = s2r_thr_copy_V.retile_D(rV);               // (16, 4, 2)

        // for(int i = 0;i < 2;i ++){
        //     for(int j = 0;j < 4;j ++){
        //         // 3. copy
        //         copy(g2s_tiled_copy_V, gV_g2s(_, _, _, i, j), sV_g2s(_, _, _, 0)); // todo mutli stage
        //         cp_async_fence();
        //         cp_async_wait<0>();
        //         __syncthreads();
        //         copy(s2r_tiled_copy_V, sV_s2r(_, _, _, 0), rV_s2r(_, _, _));
        //         // 4. compute
        //         // gemm(tiled_mma, rO(_, _, _, i), rS, rV, rO(_, _, _, i));
        //     }
        // }
    }
}

void prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor& S, torch::Tensor& O){
    // Q (seq_len, head_dim) head_dim major
    // K (seq_len, head_dim) head_dim major
    // V (seq_len, head_dim) head_dim major
    // 1.check todo
    const int seq_len = Q.size(0);
    const int head_dim = Q.size(1);
    // 2. each 4 warp cope with 128 seq length query
    dim3 griddim(seq_len / 128);
    dim3 blockdim(128);
    prefill_kernel<<<griddim, blockdim, 49152>>>(
        static_cast<half*>(Q.data_ptr()),
        static_cast<half*>(K.data_ptr()),
        static_cast<half*>(V.data_ptr()),
        static_cast<half*>(S.data_ptr()),
        static_cast<half*>(O.data_ptr()),
        seq_len, head_dim
    );
}

int main(){
    int seq_len = 1024;
    int head_dim = 256;
    auto Q = torch::randn({seq_len, head_dim}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto K = torch::randn({seq_len, head_dim}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto V = torch::randn({seq_len, head_dim}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto S = torch::zeros({seq_len, seq_len }, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto O = torch::zeros({seq_len, head_dim}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    prefill(Q, K, V, S, O);

    auto S_ref = torch::matmul(Q, K.transpose(0, 1));
    auto O_ref = torch::matmul(S_ref, V);
    // puts("-------------- S ----------------");
    // std::cout << S.slice(0, 0, 16).slice(1, 0, 128) << std::endl;
    // puts("-------------- S ref ----------------");
    // std::cout << S_ref.slice(0, 0, 16).slice(1, 0, 128) << std::endl;
    // puts("-------------- S ----------------");
    // std::cout << S.slice(1, 0, 16) << std::endl;
    // puts("-------------- S ref ----------------");
    // std::cout << S_ref.slice(1, 0, 16) << std::endl;

    // puts("-------------- S ----------------");
    // std::cout << S_ref.slice(0, 0, 128) << std::endl;
    // puts("-------------- S ref ----------------");
    // std::cout << S_ref.slice(0, 0, 128) << std::endl;

    test(S_ref.slice(1, 0, 128), S.slice(1, 0, 128), "S.slice(1, 0, 128) test");
    test(S_ref.slice(0, 0, 128), S.slice(0, 0, 128), "S.slice(0, 0, 128) test");
    test(S_ref, S, "S test");
    test(O_ref, O, "O test");

    return 0;
}