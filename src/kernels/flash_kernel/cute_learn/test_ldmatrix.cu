#include<iostream>
#include<torch/torch.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>


using namespace cute;

__global__ void kernel(half* aptr){
    __shared__ half shm_data[16 * 16];
    Tensor gA = make_tensor(make_gmem_ptr(aptr), make_shape(Int<16>{}, Int<16>{}), make_stride(Int<1>{}, Int<16>{}));
    Tensor sA = make_tensor(make_smem_ptr(shm_data), make_shape(Int<16>{}, Int<16>{}), make_stride(Int<1>{}, Int<16>{}));

    auto g2s_tiled_copy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, half>{},
                                make_layout(make_shape(Int<2>{}, Int<16>{}), make_stride(Int<1>{}, Int<2>{})),
                                make_layout(make_shape(Int<8>{}, Int<1>{})));
    auto g2s_thr_copy = g2s_tiled_copy.get_slice(threadIdx.x);
    Tensor gA_g2s = g2s_thr_copy.partition_S(gA); // (8, 1, 1)
    Tensor sA_g2s = g2s_thr_copy.partition_S(sA); // (8, 1, 1)
    copy(g2s_tiled_copy, gA_g2s(_, _, _), sA_g2s(_, _, _));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    if(thread0()){
        printf("gA: ");print(gA);printf("\n");
        printf("sA: ");print(sA);printf("\n");
    }

    auto tiled_mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), 
        Tile<Int<16>, Int<8>, Int<16>>{});
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto rA = thr_mma.partition_fragment_A(gA);
    
    auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, half>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
    auto sA_s2r = s2r_thr_copy_a.partition_S(sA);              // (CPY, CPY_M, CPY_K) = (8, 1, 1)
    auto rA_s2r = s2r_thr_copy_a.retile_D(rA);                 // (CPY, CPY_M, CPY_K) = (8, 1, 1)
    copy(s2r_tiled_copy_a, sA_s2r(_, _, _), rA_s2r(_, _, _));

    if(thread0()){
        printf("rA: ");print_tensor(rA);printf("\n");
    }
}

int main(){
    torch::Tensor a = torch::arange(16 * 16, torch::dtype(torch::kHalf).device(torch::kCUDA)).reshape({16, 16});
    puts("------------- a -----------------");
    std::cout << a << std::endl;

    puts("------------- kernel -----------------");
    kernel<<<1, 32>>>(static_cast<half*>(a.data_ptr()));
    cudaDeviceSynchronize();
}