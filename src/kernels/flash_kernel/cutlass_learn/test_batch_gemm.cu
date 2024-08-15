#include<iostream>
#include<torch/torch.h>
#include<cutlass/cutlass.h>
#include<cutlass/layout/matrix.h>
#include<cutlass/gemm/device/gemm_array.h>
#include<cutlass/gemm/device/gemm_batched.h>

void batch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor& C){
    // A (b, m, k) : (m * k, k, 1) row major
    // B (b, n, k) : (n * k, k, 1) column major
    // C (b, m, n) : (m * n, n, 1) row major
    int const batch_count = A.size(0);
    int const m = A.size(1);
    int const n = B.size(1); 
    int const k = A.size(2);
    float alpha = 1.;
    float beta = 0.;
    int const lda = k;
    int const ldb = k;
    int const ldc = n;
    int const batch_stride_A = m * k;
    int const batch_stride_B = n * k;
    int const batch_stride_C = m * n;
    auto Aptr = static_cast<float*>(A.data_ptr());
    auto Bptr = static_cast<float*>(B.data_ptr());
    auto Cptr = static_cast<float*>(C.data_ptr());
    using Gemm = cutlass::gemm::device::GemmBatched<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor
    >;
    Gemm gemm_op;
    gemm_op({
        {m, n, k},
        {Aptr, lda},
        batch_stride_A,
        {Bptr, ldb}, 
        batch_stride_B,
        {Cptr, ldc}, 
        batch_stride_C,
        {Cptr, ldc}, 
        batch_stride_C,
        {alpha, beta}, 
        batch_count
    });
}

int main(){
    const int batch_size = 16;
    const int M = 512;
    const int N = 512;
    const int K = 256;
    
    auto A = torch::randn({batch_size, M, K}, torch::dtype(torch::kFloat32).device(torch::kCUDA)); 
    auto B = torch::randn({batch_size, N, K}, torch::dtype(torch::kFloat32).device(torch::kCUDA)); 
    auto C = torch::randn({batch_size, M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)); 

    auto C_ref = torch::matmul(A, B.transpose(1, 2));
    std::cout << C_ref.sizes() << std::endl;

    batch_gemm(A, B, C);

    const float eps = 1e-2;
    if(torch::allclose(C_ref, C_ref, eps, eps))puts("passed");
    else puts("failed");

    if(torch::allclose(C, C_ref, eps, eps))puts("passed");
    else puts("failed");
    return 0;
}