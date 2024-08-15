#include"cutlass/gemm/device/gemm.h"
#include<iostream>
#include<torch/torch.h>
cudaError_t CutlassSgemmTN(int M, int N, int K, float alpha, float const* A, int lda, float const* B, int ldb, float beta, float*C, int ldc){
    using CutlassGemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor>;
    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float beta, float *C, int ldc) {
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    ColumnMajor,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix
    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({M, N, K},   // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {C, ldc},    // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}


int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 256;
    auto A = torch::randn({K, M}, torch::dtype(torch::kFloat).device(torch::kCUDA)).transpose(0, 1); // (M, K) : (1, M)
    auto B = torch::randn({N, K}, torch::dtype(torch::kFloat).device(torch::kCUDA)).transpose(0, 1); // (K, N) : (1, K)
    auto C = torch::randn({N, M}, torch::dtype(torch::kFloat).device(torch::kCUDA)).transpose(0, 1); // (M, N) : (1, M)

    // A (M, K) : (1, M) column major
    // B (N, K) : (K, 1) column major
    // C (M, N) : (1, M) column major
    int lda = M;
    int ldb = K;
    int ldc = M;
    float alpha = 1.;
    float beta = 0.;

    cudaError_t result = CutlassSgemmNN(M, N, K, alpha, static_cast<float*>(A.data_ptr()), lda, static_cast<float*>(B.data_ptr()) , ldb, beta, static_cast<float*>(C.data_ptr()), ldc);
    if(result != cudaSuccess){
        puts("CutlassSgemmNN error");
        exit(0);
    }

    auto C_ref = torch::matmul(A, B);

    std::cout << C_ref.sizes() << std::endl;

    if(torch::allclose(C_ref, C, 1e-3, 1e-3))puts("pass");
    else puts("failed");

    puts("------------- C -----------------");
    std::cout << C.slice(0, 0, 4).slice(1, 0, 4) << std::endl;
    puts("------------- C_ref -----------------");
    std::cout << C_ref.slice(0, 0, 4).slice(1, 0, 4) << std::endl;

    return 0;
}