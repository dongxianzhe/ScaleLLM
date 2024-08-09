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
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                    make_layout(make_shape(Int<4>{}, Int<1>{}, Int<1>{})),
                                    Tile<Int<64>, Int<32>, Int<16>>{});
    auto g2s_tiled_copy_qk = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto g2s_tiled_copy_v = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half>{}, 
                                make_layout(make_shape(Int<4>{}, Int<32>{}), make_stride(Int<1>{}, Int<4>{})),
                                make_layout(make_shape(Int<8>{}, Int<1>{})));
    auto s2r_tiled_copy_q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_tiled_copy_k = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half>{}, tiled_mma);
    auto s2r_tiled_copy_v = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, half>{}, tiled_mma);
    auto r2s_tiled_copy_s = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma);
    auto r2s_tiled_copy_o = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, half>{}, tiled_mma);
    auto s2g_tiled_copy_s = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half>{},
                                make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<1>{}, Int<8>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto s2g_tiled_copy_o = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half>{},
                                make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<1>{}, Int<8>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{})));
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto g2s_thr_copy_qk = g2s_tiled_copy_qk.get_slice(threadIdx.x);
    auto g2s_thr_copy_v  = g2s_tiled_copy_v.get_slice(threadIdx.x);
    auto s2r_thr_copy_q  = s2r_tiled_copy_q.get_slice(threadIdx.x);
    auto s2r_thr_copy_k  = s2r_tiled_copy_k.get_slice(threadIdx.x);
    auto s2r_thr_copy_v  = s2r_tiled_copy_v.get_slice(threadIdx.x);
    auto r2s_thr_copy_s  = r2s_tiled_copy_s.get_slice(threadIdx.x);
    auto r2s_thr_copy_o  = r2s_tiled_copy_o.get_slice(threadIdx.x);
    auto s2g_thr_copy_s  = s2g_tiled_copy_s.get_slice(threadIdx.x);
    auto s2g_thr_copy_o  = s2g_tiled_copy_o.get_slice(threadIdx.x);


    extern __shared__ half shm_data[];
    half* sQptr = shm_data;
    half* sKptr = shm_data + 128 * 32 * 3;
    half* sVptr = shm_data + 128 * 32 * 3;
    half* sSptr = shm_data + 128 * 32 * 3;
    half* sOptr = shm_data + 128 * 32 * 3;

    auto gQ = make_tensor(make_gmem_ptr(Qptr), make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{})));
    auto gK = make_tensor(make_gmem_ptr(Kptr), make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{})));
    auto gV = make_tensor(make_gmem_ptr(Vptr), make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<1>{}, Int<128>{})));
    auto gS = make_tensor(make_gmem_ptr(Sptr), make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{})));
    auto gO = make_tensor(make_gmem_ptr(Optr), make_layout(make_shape(Int<16>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{})));
    auto sQ = make_tensor(make_smem_ptr(sQptr), make_layout(make_shape(Int<128>{}, Int<32>{}, Int<3>{}), make_stride(Int<128>{}, Int<1>{}, Int<4096>{})));
    auto sK = make_tensor(make_smem_ptr(sKptr), make_layout(make_shape(Int<128>{}, Int<32>{}, Int<3>{}), make_stride(Int<128>{}, Int<1>{}, Int<4096>{})));
    auto sV = make_tensor(make_smem_ptr(sVptr), make_layout(make_shape(Int<128>{}, Int<32>{}, Int<3>{}), make_stride(Int<1>{}, Int<128>{}, Int<4096>{})));
    auto sS = make_tensor(make_smem_ptr(sVptr), make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{})));
    auto sO = make_tensor(make_smem_ptr(sVptr), make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{})));
    auto rQ = make_tensor<half>(make_layout(make_shape(Int<8>{}, Int<2 >{}, Int<2 >{})));
    auto rK = make_tensor<half>(make_layout(make_shape(Int<4>{}, Int<16>{}, Int<2 >{})));
    auto rS = make_tensor<half>(make_layout(make_shape(Int<4>{}, Int<2 >{}, Int<16>{})));

    auto gQ_tiled = local_tile(gQ, make_tile(Int<128>{}, Int<32>{}), make_coord(0, _));
    auto gK_tiled = local_tile(gQ, make_tile(Int<128>{}, Int<32>{}), make_coord(0, _));
    auto gV_tiled = local_tile(gQ, make_tile(Int<128>{}, Int<32>{}), make_coord(0, _));

    auto gQ_g2s = g2s_thr_copy_qk.partition_S(gQ_tiled); // (128, 128)   / (32, 32) -> (8, 4, 1, 4)
    auto sQ_g2s = g2s_thr_copy_qk.partition_D(sQ); // (128, 32, 3) / (32, 32) -> (8, 4, 1, 3)
    auto sQ_s2r = s2r_thr_copy_q.partition_S(sQ);  // (128, 32, 3) / (64, 16) -> (8, 2, 2, 3) 
    auto rQ_s2r = s2r_thr_copy_q.retile_D(rQ);     // (8, 2, 2)

    auto gK_g2s = g2s_thr_copy_qk.partition_S(gK_tiled); // (128, 128)   / (32, 32) -> (8, 4, 1, 4)
    auto sK_g2s = g2s_thr_copy_qk.partition_D(sK); // (128, 32, 3) / (32, 32) -> (8, 4, 1, 3)
    auto sK_s2r = s2r_thr_copy_k.partition_S(sK);  // (128, 32, 3) / (32, 16) -> (16, 4, 2, 3) 
    auto rK_s2r = s2r_thr_copy_k.retile_D(rK);     // (16, 4, 2)

    for(int i = 0;i < 8;i ++){
        if(i == 0){
            for(int k = 0;k < 2;k ++){
                copy(g2s_tiled_copy_qk, gQ_g2s(_, _, _, k), sQ_g2s(_, _, _, k));
                copy(g2s_tiled_copy_qk, gK_g2s(_, _, _, k), sK_g2s(_, _, _, k));
                cp_async_fence();
            }
            cp_async_wait<1>();
            __syncthreads();
        }
        if(i % 2 == 0){
            int itile = i / 2 + 2;
            if(itile < 8){
                copy(g2s_tiled_copy_qk, gQ_g2s(_, _, _, ), sQ_g2s(_, _, _, k));
                copy(g2s_tiled_copy_qk, gK_g2s(_, _, _, ), sK_g2s(_, _, _, k));
            }
            
        }
        if(i % 2 == 1){
            cp_async_wait<1>();
            __syncthreads();
        }
    }
}

int main(){
    auto Q = torch::randn({128, 128}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto K = torch::randn({128, 128}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto V = torch::randn({128, 128}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto S = torch::randn({128, 128}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto O = torch::randn({128, 128}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    auto S_ref = torch::matmul(Q, K.transpose(0, 1));
    auto O_ref = torch::matmul(S_ref, V);

    kernel<<<1, 128, 48 * 1024>>>(
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
    test(O_ref, O, "O test");

}