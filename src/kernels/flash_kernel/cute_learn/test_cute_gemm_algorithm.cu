#include<iostream>
#include<torch/torch.h>
#include<cute/tensor.hpp>
#include<cute/algorithm/gemm.hpp>
using namespace cute;

// __global__ void kernel1(float* aptr, float* bptr, float* cptr, int n){
//     Tensor a = make_tensor(make_gmem_ptr(aptr), make_layout(make_shape(n), make_stride(1)));
//     Tensor b = make_tensor(make_gmem_ptr(bptr), make_layout(make_shape(n), make_stride(1)));
//     Tensor c = make_tensor(make_gmem_ptr(cptr), make_layout(make_shape(n), make_stride(1)));

//     cute::gemm(a, b, c);
//     // if(thread0())print_tensor(c);
// }

int main(){
    // puts("1. (V) x (V) => (V). The element-wise product of vectors: Cv += Av Bv. Dispatches to FMA or MMA.");
    // {
    //     const int n = 32;
    //     torch::Tensor ta = torch::arange(n, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //     torch::Tensor tb = torch::arange(n, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //     torch::Tensor tc = torch::empty (n, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //     float* aptr = static_cast<float*>(ta.data_ptr());
    //     float* bptr = static_cast<float*>(tb.data_ptr());
    //     float* cptr = static_cast<float*>(tc.data_ptr());

    //     kernel1<<<1, 32>>>(aptr, bptr, cptr,n);
    //     Tensor c = make_tensor(cptr, make_shape(n));
    //     print_tensor(c);
    // }

    // (M) x (N) => (M,N). The outer product of vectors: Cmn += Am B_n. Dispatches to (4) with V=1.
    // (M,K) x (N,K) => (M,N). The product of matrices: Cmn += Amk Bnk. Dispatches to (2) for each K.
    // (V,M) x (V,N) => (V,M,N). The batched outer product of vectors: Cvmn += Avm Bvn. Optimizes for register reuse and dispatches to (1) for each M, N.
    // (V,M,K) x (V,N,K) => (V,M,N). The batched product of matrices: Cvmn += Avmk Bvnk. Dispatches to (4) for each K.
}