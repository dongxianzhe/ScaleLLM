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
__global__ void kernel(half* Qptr, half* Kptr, half* Vptr, half* Sptr, half* Optr){
    using namespace cute;
    auto gQ = make_tensor(make_gmem_ptr(Qptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    auto gK = make_tensor(make_gmem_ptr(Kptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    auto gV = make_tensor(make_gmem_ptr(Vptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<1>{}, Int<16>{})));
    auto gS = make_tensor(make_gmem_ptr(Sptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    auto gO = make_tensor(make_gmem_ptr(Optr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));

    __shared__ half sQptr[16 * 16];
    __shared__ half sKptr[16 * 16];
    auto sQ = make_tensor(make_smem_ptr(sQptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    auto sK = make_tensor(make_smem_ptr(sKptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));

    auto g2s_tiled_copy_QK = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                        make_layout(make_shape(Int<16>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{})),
                                        make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto g2s_thr_copy_QK = g2s_tiled_copy_QK.get_slice(threadIdx.x);
    auto gQ_g2s = g2s_thr_copy_QK.partition_S(gQ); // (8, 1, 1)
    auto sQ_g2s = g2s_thr_copy_QK.partition_D(sQ); // (8, 1, 1)
    auto gK_g2s = g2s_thr_copy_QK.partition_S(gK); // (8, 1, 1)
    auto sK_g2s = g2s_thr_copy_QK.partition_D(sK); // (8, 1, 1)
    copy(g2s_tiled_copy_QK, gQ_g2s, sQ_g2s);
    copy(g2s_tiled_copy_QK, gK_g2s, sK_g2s);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                    make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // 这个决定mma的tile
                                    Tile<Int<16>, Int<16>, Int<16>>{}); // 这个决定copy的tile
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto rQ = thr_mma.partition_fragment_A(gQ); // (8, 1, 1)
    auto rK = thr_mma.partition_fragment_B(gK); // (4, 2, 1)
    auto rS = thr_mma.partition_fragment_C(gS); // (4, 1, 2)

    auto s2r_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_thr_copy_Q = s2r_tiled_copy_Q.get_slice(threadIdx.x);
    auto sQ_s2r = s2r_thr_copy_Q.partition_S(sQ); // (8, 1, 1)
    auto rQ_s2r = s2r_thr_copy_Q.retile_D(rQ);    // (8, 1, 1)
    copy(s2r_tiled_copy_Q, sQ_s2r, rQ_s2r);

    auto s2r_tiled_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_thr_copy_K = s2r_tiled_copy_K.get_slice(threadIdx.x);
    auto sK_s2r = s2r_thr_copy_K.partition_S(sK); // (8, 1, 1)
    auto rK_s2r = s2r_thr_copy_K.retile_D(rK);    // (8, 1, 1)
    copy(s2r_tiled_copy_K, sK_s2r, rK_s2r);

    gemm(tiled_mma, rS, rQ, rK, rS);

    __shared__ half sSptr[16 * 16];
    auto sS = make_tensor(make_smem_ptr(sSptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    auto r2s_tiled_copy_S = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma);
    auto r2s_thr_copy_S = r2s_tiled_copy_S.get_slice(threadIdx.x);
    auto rS_s2r = r2s_thr_copy_S.retile_S(rS);    // (8, 1, 1)
    auto sS_s2r = r2s_thr_copy_S.partition_D(sS); // (8, 1, 1)
    copy(r2s_tiled_copy_S, rS_s2r, sS_s2r);

    auto s2g_tiled_copy_S = make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, half>{},
                                            make_layout(make_shape(Int<16>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{})),
                                            make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto s2g_thr_copy_S = s2g_tiled_copy_S.get_slice(threadIdx.x);
    auto sS_s2g = s2g_thr_copy_S.partition_S(sS); // (8, 1, 1)
    auto gS_s2g = s2g_thr_copy_S.partition_D(gS); // (8, 1, 1)
    copy(s2g_tiled_copy_S, sS_s2g, gS_s2g);

    __shared__ half sVptr[16 * 16];
    auto sV = make_tensor(make_smem_ptr(sVptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<1>{}, Int<16>{})));
    auto g2s_tiled_copy_V = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                        make_layout(make_shape(Int<2>{}, Int<16>{}), make_stride(Int<1>{}, Int<2>{})),
                                        make_layout(make_shape(Int<8>{}, Int<1>{})));
    auto g2s_thr_copy_V = g2s_tiled_copy_V.get_slice(threadIdx.x);
    auto gV_g2s = g2s_thr_copy_V.partition_S(gV); // (8, 1, 1)
    auto sV_g2s = g2s_thr_copy_V.partition_D(sV); // (8, 1, 1)

    copy(g2s_tiled_copy_V, gV_g2s, sV_g2s);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    
    auto rS2 = make_tensor(rS.data(), make_layout(make_shape(Int<8>{}, Int<1>{}, Int<1>{}), make_stride(Int<1>{}, Int<0>{}, Int<0>{})));
    auto rV = thr_mma.partition_fragment_B(gV); // (4, 2, 1)
    auto rO = thr_mma.partition_fragment_C(gO); // (4, 1, 2)

    auto s2r_tiled_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, half>{}, tiled_mma);
    auto s2r_thr_copy_V = s2r_tiled_copy_V.get_slice(threadIdx.x);
    auto sV_s2r = s2r_thr_copy_V.partition_S(sV);
    auto rV_s2r = s2r_thr_copy_V.retile_D(rV);

    copy(s2r_tiled_copy_V, sV_s2r, rV_s2r);

    gemm(tiled_mma, rO, rS2, rV, rO);

    // if(thread0()){
        // printf(": rS ");print_tensor(rS);printf("\n");
    //     printf(": rS2");print_tensor(rS2);printf("\n");
    //     printf(": rV ");print_tensor(rV);printf("\n");
    //     printf("rO: ");print_tensor(rO);printf("\n");
    // }

    __shared__ half sOptr[16 * 16];
    auto sO = make_tensor(make_smem_ptr(sOptr), make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    auto r2s_tiled_copy_O = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma);
    auto r2s_thr_copy_O = r2s_tiled_copy_O.get_slice(threadIdx.x);
    auto rO_r2s = r2s_thr_copy_O.retile_S(rO);    // (8, 1, 1)
    auto sO_r2s = r2s_thr_copy_O.partition_D(sO); // (8, 1, 1)
    copy(r2s_tiled_copy_O, rO_r2s, sO_r2s);

    auto s2g_tiled_copy_O = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half>{}, 
                                            make_layout(make_shape(Int<16>{}, Int<2>{}), make_stride(Int<2>{}, Int<1>{})),
                                            make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto s2g_thr_copy_O = s2g_tiled_copy_O.get_slice(threadIdx.x);
    auto sO_g2s = s2g_thr_copy_O.partition_S(sO); // (8, 1, 1)
    auto gO_g2s = s2g_thr_copy_O.partition_D(gO); // (8, 1, 1)
    copy(s2g_tiled_copy_O, sO_g2s, gO_g2s);
}

int main(){
    auto Q = torch::randn({16, 16}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto K = torch::randn({16, 16}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto V = torch::randn({16, 16}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto S = torch::randn({16, 16}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto O = torch::randn({16, 16}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    auto S_ref = torch::matmul(Q, K.transpose(0, 1));
    auto O_ref = torch::matmul(S_ref, V);

    kernel<<<1, 32>>>(
        static_cast<half*>(Q.data_ptr()),
        static_cast<half*>(K.data_ptr()),
        static_cast<half*>(V.data_ptr()),
        static_cast<half*>(S.data_ptr()),
        static_cast<half*>(O.data_ptr())
    );
    // puts("------------ S_ref ------------------");
    // std::cout << S_ref << std::endl;
    // puts("------------ S ------------------");
    // std::cout << S << std::endl;
    // puts("---------- O_ref --------------------");
    // std::cout << O_ref << std::endl;
    // puts("---------- O --------------------");
    // std::cout << O << std::endl;

    test(S_ref, S, "S test");
    test(O_ref, O, "S test");

}