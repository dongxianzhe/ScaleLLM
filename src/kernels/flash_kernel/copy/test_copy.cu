#include<iostream>
#include<torch/torch.h>
#include<cute/stride.hpp>
#include<cute/layout.hpp>
#include<cute/tensor.hpp>
using namespace cute;

constexpr int M = 32;
constexpr int K = 8;


#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
__global__ void copy_naive_kernel(half* a, half* b){
    int tid = threadIdx.x;
    __shared__ half s[M][K];
    (reinterpret_cast<float4*> (&s[tid][0]))[0] = (reinterpret_cast<float4*> (&a[tid * 8]))[0];
    (reinterpret_cast<float4*> (&b[tid * 8]))[0] = (reinterpret_cast<float4*> (&s[tid][0]))[0];
}

torch::Tensor copy_naive(torch::Tensor a){
    torch::Tensor b = torch::empty({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    dim3 gridDim(1);
    dim3 blockDim(32);
    copy_naive_kernel<<<gridDim, blockDim>>>(
        static_cast<half*>(a.data_ptr()), 
        static_cast<half*>(b.data_ptr())
        );
    return b;
}

__global__ void copy_async_kernel(half* a, half* b){
    int tid = threadIdx.x;
    __shared__ half s[M][K];
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, half>;
    using G2SCopy = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{})),
        make_layout(make_shape(Int<8>{}))
        ));

    if(thread0()){
        print(G2SCopy{});printf("\n");
    }

    Tensor ga = make_tensor(make_gmem_ptr(a), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    Tensor sa = make_tensor(make_smem_ptr(s), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    // Tensor gb = make_tensor(make_gmem_ptr(b), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    G2SCopy g2scopy;
    ThrCopy thrcopy = g2scopy.get_thread_slice(tid);
    Tensor taga_copy = thrcopy.partition_S(ga);
    if(thread0()){
        print(taga_copy);
    }

    // (reinterpret_cast<float4*> (&b[tid * 8]))[0] = (reinterpret_cast<float4*> (&s[tid][0]))[0];
}

torch::Tensor copy_async(torch::Tensor a){
    torch::Tensor b = torch::empty({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));

    dim3 gridDim(1);
    dim3 blockDim(32);
    copy_async_kernel<<<gridDim, blockDim>>>(
        static_cast<half*>(a.data_ptr()), 
        static_cast<half*>(b.data_ptr())
        );

    return b;
}

void test(torch::Tensor o, torch::Tensor o_ref,const char* name){
    if(o.is_cuda())o = o.to(torch::kCPU);
    if(o_ref.is_cuda())o_ref = o_ref.to(torch::kCPU);

    const float atol = 1e-1;
    const float rtol = 1e-1;

    if(torch::allclose(o, o_ref, atol, rtol))printf("%s pass\n", name);
    else printf("%s fail\n", name);
}

int main(){
    torch::Tensor a  = torch::randn({M, K}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    torch::Tensor b1 = copy_naive(a);
    torch::Tensor b2 = copy_async(a);

    test(a, b1, "copy_naive");
    test(a, b2, "copy_async");
}