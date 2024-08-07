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

namespace prefill_kernel_config{
    using namespace cute;

    // using MMA = decltype(make_tiled_mma(
    //     SM80_16x8x16_F16F16F16F16_TN{},
    //     make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})), 
    //     Tile<Int<32>, Int<32>, Int<16>>{}));
    using MMA = decltype(make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})), 
        Tile<Int<64>, Int<32>, Int<16>>{}));

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


    constexpr int kSmemLayoutSBatch = 2;
    using SmemLayoutAtomS = decltype(composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}))));
    using SmemLayoutS = decltype(tile_to_shape(
        SmemLayoutAtomS{},
        make_shape(Int<32>{}, Int<32>{}, Int<kSmemLayoutSBatch>{})));
};

__global__ void prefill_kernel(half* Qptr, half* Kptr, half* Vptr, half* Sptr, int seq_len, int head_dim){
    using namespace cute;
    using namespace prefill_kernel_config;
    extern __shared__ half shm_data[];
    // 1. make q k v tensor
    Tensor Q = make_tensor(make_gmem_ptr(Qptr), make_shape(seq_len, head_dim), make_stride(head_dim, Int<1>{}));
    Tensor K = make_tensor(make_gmem_ptr(Kptr), make_shape(seq_len, head_dim), make_stride(head_dim, Int<1>{}));
    Tensor V = make_tensor(make_gmem_ptr(Vptr), make_shape(seq_len, head_dim), make_stride(head_dim, Int<1>{}));
    Tensor S = make_tensor(make_gmem_ptr(Sptr), make_shape(seq_len, seq_len ), make_stride(seq_len, Int<1>{}));

    // 2. divide q k
    Tensor gQ = local_tile(Q, make_tile(Int<128>{}, Int<32>{}), make_coord(blockIdx.x, _)); // (128, 32, 8)
    Tensor gK = local_tile(K, make_tile(Int<128>{}, Int<32>{}), make_coord(_, _));          // (128, 32, 8, 8) = (128, 32, 1024 / 128, 256 / 32)
    Tensor gS = local_tile(S, make_tile(Int<128>{}, Int<128>{}), make_coord(blockIdx.x, _));// (128, 128, 8) = (128, 128, 1024 / 128)

    // if(thread0()){
    //     printf("gQ: ");print(gQ);printf("\n"); // (_128,_32,8):(256,_1,_32)
    //     printf("gK: ");print(gK);printf("\n"); // (_128,_32,8,8):(256,_1,32768,_32)
    //     printf("gS: ");print(gS);printf("\n"); // (_128,_128,8):(1024,_1,_128)
    // }

    // 3. Q K g2s
    Tensor sQ = make_tensor(make_smem_ptr(shm_data), SmemLayoutQ{}); 
    Tensor sK = make_tensor(make_smem_ptr(shm_data + cute::cosize(SmemLayoutQ{})), SmemLayoutK{});

    auto g2s_tiled_copy_ab = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto g2s_thr_copy_ab = g2s_tiled_copy_ab.get_slice(threadIdx.x);
    auto tAgQ_copy = g2s_thr_copy_ab.partition_S(gQ);  // (CPY, CPY_M, CPY_K, k)      = (8, 4, 1, 8) = (8, 128 / 32, 32 / 32, 256 / 32)
    auto tAsQ_copy = g2s_thr_copy_ab.partition_D(sQ);  // (CPY, CPY_M, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    auto tBgK_copy = g2s_thr_copy_ab.partition_S(gK);  // (CPY, CPY_N, CPY_K, n, k)      = (8, 4, 1, 8, 8) = (8, 128 / 32, 32 / 32, 1024 / 128, 256 / 32)
    auto tBsK_copy = g2s_thr_copy_ab.partition_D(sK);  // (CPY, CPY_N, CPY_K, kStage) = (8, 4, 1, 3) = (8, 128 / 32, 32 / 32, 3)
    // if(thread0()){
    //     printf("sQ: ");print(sQ);printf("\n"); // Sw<3,3,3> o _0 o (_128,_32,_3):(_32,_1,_4096)
    //     printf("sK: ");print(sK);printf("\n"); // Sw<3,3,3> o _0 o (_128,_32,_3):(_32,_1,_4096)
    //     printf("tAgQ_copy: ");print(tAgQ_copy);printf("\n"); // ((_8,_1),_4,_1,8):((_1,_0),8192,_0,_32)
    //     printf("tAsQ_copy: ");print(tAsQ_copy);printf("\n"); // ((_8,_1),_4,_1,_3):((_1,_0),_1024,_0,_4096)
    //     printf("tBgK_copy: ");print(tBgK_copy);printf("\n"); // ((_8,_1),_4,_1,8,8):((_1,_0),8192,_0,32768,_32)
    //     printf("tBsK_copy: ");print(tBsK_copy);printf("\n"); // ((_8,_1),_4,_1,_3):((_1,_0),_1024,_0,_4096)
    // }

    // 4. mma
    MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrQ = thr_mma.partition_fragment_A(gQ(_, _, 0));     // (MMA, MMA_M, MMA_K) = (8, 2, 2) = (8, 128 / 64, 32 / 16)
    auto tCrK = thr_mma.partition_fragment_B(gK(_, _, 0, 0));  // (MMA, MMA_N, MMA_K) = (4, 16, 2) = (4, 128 / 8, 32 / 16)
    auto tCrS = thr_mma.partition_fragment_C(gS(_, _, 0));     // (MMA, MMA_M, MMA_N) = (4, 2, 16) = (4, 128 / 64, 128 / 8)
    // if(thread0()){
    //     printf("tCrQ: ");print(tCrQ);printf("\n"); // ((_2,_2,_2),_2,_2):((_1,_2,_4),_16,_8)
    //     printf("tCrK: ");print(tCrK);printf("\n"); // ((_2,_2),(_4,_4),_2):((_1,_2),(_8,_32),_4)
    //     printf("tCrS: ");print(tCrS);printf("\n"); // ((_2,_2),_2,_16):((_1,_2),_4,_8)
    // }
    // 5. Q K s2r
    auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    auto tAsQ = s2r_thr_copy_a.partition_S(sQ);                   // (CPY, CPY_M, CPY_K, kStage) = (8, 2, 2, 3)
    auto tCrQ_view = s2r_thr_copy_a.retile_D(tCrQ);               // (CPY, CPY_M, CPY_K)         = (8, 2, 2)

    auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(threadIdx.x);
    auto tBsK = s2r_thr_copy_b.partition_S(sK);                   // (CPY, CPY_M, CPY_K, kStage) = (16, 4, 2, 3)
    auto tCrK_view = s2r_thr_copy_b.retile_D(tCrK);               // (CPY, CPY_M, CPY_K)         = (16, 4, 2)
    // if(thread0()){
    //     printf("tAsQ     : ");print(tAsQ);     printf("\n");// ((_8,_1),_2,_2,_3):((_1,_0),_2048,16,_4096)
    //     printf("tCrQ_view: ");print(tCrQ_view);printf("\n");// ((_8,_1),_2,_2):((_1,_0),_16,_8)
    //     printf("tBsK     : ");print(tBsK);     printf("\n");// ((_8,_2),_4,_2,_3):((_1,_512),_1024,16,_4096)
    //     printf("tCrK_view: ");print(tCrK_view);printf("\n");// (((_4,_2),_2),_4,_2):(((_1,_8),_16),_32,_4)
    // }

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
    // copy S back to test S
    auto sS = make_tensor(make_smem_ptr(shm_data), make_layout(make_shape(Int<64>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})));
    auto r2s_tiled_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma); // (64, 32)
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x); // CPY = 16
    auto tCrS_r2s = r2s_thr_copy_c.retile_S(tCrS);   // (CPY, CPY_M, CPY_N) = (16, 2, 4) // (16, 128 / 64, 128 / 32)
    auto tCsS_r2s = r2s_thr_copy_c.partition_D(sS);  // (CPY, _1, _1) = (16, 1, 1)

    auto s2g_tiled_copy_c = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, half>{},
                            make_layout(make_shape(Int<32>{}, Int<4>{}),
                                        make_stride(Int<4>{}, Int<1>{})),
                            make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(threadIdx.x);
    auto tCsS_s2g = s2g_thr_copy_c.partition_S(sS);  // (CPY, CPY_M, CPY_N) = (8, 2, 1)
    auto tCgS_s2g = s2g_thr_copy_c.partition_D(gS);  // (CPY, CPY_M, CPY_N) = (8, 4, 4, 8) = (8, 128 / 32, 128 / 32, 1024 / 128)
    for(int i = 0;i < 2;i ++){
        for(int j = 0;j < 4;j ++){
            copy(r2s_tiled_copy_c, tCrS_r2s(_, i, j), tCsS_r2s(_, 0, 0));
            __syncthreads();
            copy(s2g_tiled_copy_c, tCsS_s2g(_, 0, j), tCgS_s2g(_, i * 2, j, 0));
            copy(s2g_tiled_copy_c, tCsS_s2g(_, 1, j), tCgS_s2g(_, i * 2 + 1, j, 0));
            __syncthreads();
        }
    }
    // auto sS = make_tensor(make_smem_ptr(shm_data), SmemLayoutS{});
    // auto r2s_tiled_copy_c = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma);
    // auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x);
    // auto tCrS_r2s = r2s_thr_copy_c.retile_S(tCrS);   // (CPY, CPY_M, CPY_N) = (16, 2, 4) // (16, 128 / 64, 128 / 32)
    // auto tCsS_r2s = r2s_thr_copy_c.partition_D(sS);  // (CPY, _1, _1, pipe) = (8, 1, 1, 2)
    // auto s2g_tiled_copy_c = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, half>{},
    //                         make_layout(make_shape(Int<32>{}, Int<4>{}),
    //                                     make_stride(Int<4>{}, Int<1>{})),
    //                         make_layout(make_shape(Int<1>{}, Int<8>{})));
    // auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(threadIdx.x);
    // auto tCsS_s2g = s2g_thr_copy_c.partition_S(sS);  // (CPY, CPY_M, CPY_N, pipe) = (8, 1, 1, 2)
    // auto tCgS_s2g = s2g_thr_copy_c.partition_D(gS);  // (CPY, CPY_M, CPY_N) = (8, 4, 4, 8) = (8, 128 / 32, 128 / 32, 1024 / 128)
    // if(thread0()){
    //     printf("tCrS_r2s: ");print(tCrS_r2s);printf("\n"); // ((_2,(_2,_4)),_2,_4):((_1,(_2,_8)),_4,_32)
    //     printf("tCsS_r2s: ");print(tCsS_r2s);printf("\n"); // ((_2,(_2,(_2,_2))),_1,_1,_2):((_1,(_256,(8,16))),_0,_0,_1024)
    //     printf("tCsS_s2g: ");print(tCsS_s2g);printf("\n"); // ((_8,_1),_1,_1,_2):((_1,_0),_0,_0,_1024)
    //     printf("tCgS_s2g: ");print(tCgS_s2g);printf("\n"); // ((_8,_1),_4,_4,8):((_1,_0),32768,_32,_128)
    // }
    // auto tCgS_s2gx = group_modes<1, 3>(tCgS_s2g);  // (CPY_, CPY_MN) = (8, 16) 一个tiled copy指令32乘32，每个线程需要8个half，M复制4次，N复制四次
    // auto tCrS_r2sx = group_modes<1, 3>(tCrS_r2s);  // (CPY_, CPY_MN) = (16, 8) 一个tiledmma 64x32的C每个寄存器需要16个寄存器，M复制2次，N复制4次

    //     int step = size<3>(tCsS_r2s); // 2
    // #pragma unroll
    //     for (int i = 0; i < size<1>(tCrS_r2sx) * 2; i += step) { // i = 0, 2, 4, 6, ... , 14
    //         // reg->smem
    // #pragma unroll
    //         for (int j = 0; j < step; ++j) { // 2 
    //             auto t = make_tensor_like<half>(tCrS_r2sx(_, i + j));
    //             cute::copy(tCrS_r2sx(_, i + j), t);
    //             cute::copy(r2s_tiled_copy_c, t, tCsS_r2s(_, 0, 0, j));
    //         }
    //         __syncthreads();
    //         // smem -> global
    // #pragma unroll
    //         for (int j = 0; j < step; ++j) cute::copy(s2g_tiled_copy_c, tCsS_s2g(_, 0, 0, j), tCgS_s2gx(_, i + j, 0));
    //         __syncthreads();
    //     }
}

void prefill(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor& S){
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
    prefill(Q, K, V, S);

    auto S_ref = torch::matmul(Q, K.transpose(0, 1));
    // puts("-------------- S ----------------");
    // std::cout << S.slice(0, 0, 16).slice(1, 0, 128) << std::endl;
    // puts("-------------- S ref ----------------");
    // std::cout << S_ref.slice(0, 0, 16).slice(1, 0, 128) << std::endl;
    // puts("-------------- S ----------------");
    // std::cout << S.slice(1, 0, 16) << std::endl;
    // puts("-------------- S ref ----------------");
    // std::cout << S_ref.slice(1, 0, 16) << std::endl;

    test(S_ref.slice(1, 0, 128), S.slice(1, 0, 128), "S test");

    return 0;
}