#include<iostream>
#include<torch/torch.h>
#include<cute/stride.hpp>
#include<cute/layout.hpp>
#include<cute/tensor.hpp>
using namespace cute;


// #define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// __global__ void copy_naive_kernel(half* a, half* b){
//     int tid = threadIdx.x;
//     __shared__ half s[M][K];
//     (reinterpret_cast<float4*> (&s[tid][0]))[0] = (reinterpret_cast<float4*> (&a[tid * 8]))[0];
//     (reinterpret_cast<float4*> (&b[tid * 8]))[0] = (reinterpret_cast<float4*> (&s[tid][0]))[0];
// }

// torch::Tensor copy_naive(torch::Tensor a){
//     torch::Tensor b = torch::empty({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));

//     dim3 gridDim(1);
//     dim3 blockDim(32);
//     copy_naive_kernel<<<gridDim, blockDim>>>(
//         static_cast<half*>(a.data_ptr()), 
//         static_cast<half*>(b.data_ptr())
//         );
//     return b;
// }

// __global__ void copy_async_kernel(half* a, half* b){
//     int tid = threadIdx.x;
//     __shared__ half s[M][K];
//     using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
//     using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
//     using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half>;
//     using G2SCopy = decltype(make_tiled_copy(
//         g2s_copy_atom{},
//         make_layout(make_shape(Int<32>{})),
//         make_layout(make_shape(Int<8>{}))
//         ));

//     if(thread0()){
//         print(G2SCopy{});printf("\n");
//     }

//     Tensor ga = make_tensor(make_gmem_ptr(a), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
//     Tensor sa = make_tensor(make_smem_ptr(s), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
//     // Tensor gb = make_tensor(make_gmem_ptr(b), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
//     G2SCopy g2scopy;
//     ThrCopy thrcopy = g2scopy.get_thread_slice(tid);
//     Tensor taga_copy = thrcopy.partition_S(ga);
//     if(thread0()){
//         print(taga_copy);
//     }

//     // (reinterpret_cast<float4*> (&b[tid * 8]))[0] = (reinterpret_cast<float4*> (&s[tid][0]))[0];
// }

// torch::Tensor copy_async(torch::Tensor a){
//     torch::Tensor b = torch::empty({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));

//     dim3 gridDim(1);
//     dim3 blockDim(32);
//     copy_async_kernel<<<gridDim, blockDim>>>(
//         static_cast<half*>(a.data_ptr()), 
//         static_cast<half*>(b.data_ptr())
//         );

//     return b;
// }

void test(torch::Tensor o, torch::Tensor o_ref,const char* name){
    if(o.is_cuda())o = o.to(torch::kCPU);
    if(o_ref.is_cuda())o_ref = o_ref.to(torch::kCPU);

    const float atol = 1e-1;
    const float rtol = 1e-1;

    if(torch::allclose(o, o_ref, atol, rtol))printf("%s pass\n", name);
    else printf("%s fail\n", name);
}

const int M = 1024;
const int K = 256;
const int BM = 128;
const int BK = 8;

__global__ void copy_v1_kernel(float* aptr, float* bptr){
    using namespace cute;
    Tensor a = make_tensor(make_gmem_ptr(aptr), make_layout(make_shape(M, K), make_stride(K, 1)));
    Tensor b = make_tensor(make_gmem_ptr(bptr), make_layout(make_shape(M, K), make_stride(K, 1)));
    // ga gb
    Tensor ga = local_tile(a, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.x, _)); // ((BM, BK), (NBM, NBK)) -> (BM, BK, NBK)
    Tensor gb = local_tile(b, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.x, _));
    // ta
    Layout ta = make_layout(Int<32>{}, Int<8>{});
    // taga
    Tensor taga = local_partition(ga, ta, threadIdx.x); // ((tm, tk), (ntm, ntk, NBK)) -> (ntm, ntk, NBK)
    Tensor tagb = local_partition(gb, ta, threadIdx.x); // ((tm, tk), (ntm, ntk, NBK)) -> (ntm, ntk, NBK)
    // if(thread0()){
    //     printf("a : ");print(a);printf("\n");
    //     printf("ga: ");print(ga);printf("\n");
    //     printf("ta: ");print(ta);printf("\n");
    //     printf("taga: ");print_tensor(taga);printf("\n");
    // }

    for(int k = 0;k < K / BK;k ++){
        copy(taga(_, _, k), tagb(_, _, k));
    }
}

torch::Tensor copy_v1(torch::Tensor a){
    torch::Tensor b = torch::empty({M * K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    copy_v1_kernel<<<M / BM, 32 * 8>>>(static_cast<float*>(a.data_ptr()), static_cast<float*>(b.data_ptr()));
    return b;
}

__global__ void copy_v2_kernel(float* aptr, float* bptr){
    using namespace cute;
    __shared__ float smem[BM * BK];
    Tensor a = make_tensor(make_gmem_ptr(aptr), make_layout(make_shape(M, K), make_stride(K, 1)));
    Tensor b = make_tensor(make_gmem_ptr(bptr), make_layout(make_shape(M, K), make_stride(K, 1)));
    // ga gb sa
    Tensor ga = local_tile(a, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.x, _)); // ((BM, BK), (NBM, NBK)) -> (BM, BK, NBK)
    Tensor gb = local_tile(b, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.x, _));
    Tensor sa = make_tensor(make_smem_ptr(smem), make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{})));
    // ta
    Layout ta = make_layout(Int<32>{}, Int<8>{});
    // taga
    Tensor taga = local_partition(ga, ta, threadIdx.x); // ((tm, tk), (ntm, ntk, NBK)) -> (ntm, ntk, NBK)
    Tensor tagb = local_partition(gb, ta, threadIdx.x); // ((tm, tk), (ntm, ntk, NBK)) -> (ntm, ntk, NBK)
    // tasa
    Tensor tasa = local_partition(sa, ta, threadIdx.x); // ((tm, tk), (ntm, ntk))

    // if(thread0()){
    //     printf("a : ");print(a);printf("\n");
    //     printf("ga: ");print(ga);printf("\n");
    //     printf("sa: ");print(sa);printf("\n");
    //     printf("ta: ");print(ta);printf("\n");
    //     printf("taga: ");print(taga);printf("\n");
    // }

    for(int k = 0;k < K / BK;k ++){
        copy(taga(_, _, k), tasa(_, _));

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        copy(tasa(_, _), tagb(_, _, k));

    }
}

torch::Tensor copy_v2(torch::Tensor a){
    torch::Tensor b = torch::empty({M * K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    copy_v2_kernel<<<M / BM, 32 * 8>>>(static_cast<float*>(a.data_ptr()), static_cast<float*>(b.data_ptr()));
    return b;
}

__global__ void copy_v3_kernel(float* aptr, float* bptr){
    using namespace cute;
    __shared__ float smem[BM * BK];
    Tensor a = make_tensor(make_gmem_ptr(aptr), make_layout(make_shape(M, K), make_stride(K, 1)));
    Tensor b = make_tensor(make_gmem_ptr(bptr), make_layout(make_shape(M, K), make_stride(K, 1)));
    // ga gb sa
    Tensor ga = local_tile(a, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.x, _)); // ((BM, BK), (NBM, NBK)) -> (BM, BK, NBK)
    Tensor gb = local_tile(b, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.x, _));
    Tensor sa = make_tensor(make_smem_ptr(smem), make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{})));
    // ta
    Layout ta = make_layout(Int<32>{}, Int<8>{});
    // taga
    Tensor taga = local_partition(ga, ta, threadIdx.x); // ((tm, tk), (ntm, ntk, NBK)) -> (ntm, ntk, NBK)
    Tensor tagb = local_partition(gb, ta, threadIdx.x); // ((tm, tk), (ntm, ntk, NBK)) -> (ntm, ntk, NBK)
    // tasa
    Tensor tasa = local_partition(sa, ta, threadIdx.x); // ((tm, tk), (ntm, ntk))

    // if(thread0()){
    //     printf("a : ");print(a);printf("\n");
    //     printf("ga: ");print(ga);printf("\n");
    //     printf("sa: ");print(sa);printf("\n");
    //     printf("ta: ");print(ta);printf("\n");
    //     printf("taga: ");print(taga);printf("\n");
    // }
    // using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    // using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    // using g2s_copy_atom = Copy_Atom<g2s_copy_traits, float>;
    // using G2SCopyA =
    //     decltype(make_tiled_copy(g2s_copy_atom{},
    //                             make_layout(make_shape(Int<32>{}, Int<8>{}),
    //                                         make_stride(Int<8>{}, Int<1>{})),
    //                             make_layout(make_shape(Int<1>{}, Int<8>{}))));

    for(int k = 0;k < K / BK;k ++){
        copy(taga(_, _, k), tasa(_, _));

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        copy(tasa(_, _), tagb(_, _, k));

    }
}

torch::Tensor copy_v3(torch::Tensor a){
    torch::Tensor b = torch::empty({M * K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    copy_v3_kernel<<<M / BM, 32 * 8>>>(static_cast<float*>(a.data_ptr()), static_cast<float*>(b.data_ptr()));
    return b;
}

int main(){
    torch::Tensor a = torch::arange(M * K, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor b1 = copy_v1(a);
    torch::Tensor b2 = copy_v2(a);
    test(a, b1, "copy_v1");
    test(a, b2, "copy_v2");

    // torch::Tensor b1 = copy_naive(a);
    // torch::Tensor b2 = copy_async(a);

    // test(a, b1, "copy_naive");
    // test(a, b2, "copy_async");
}