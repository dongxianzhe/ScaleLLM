#include"cutlass/gemm/device/gemm.h"
#include<torch/torch.h>

torch::Tensor CutlassSgemmTN(torch::Tensor A, torch::Tensor B){
    //        A (M, K) : (K, 1) row major
    //        B (N, K) : (K, 1) column major
    // return C (M, N) : (N, 1) row major
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    float alpha = 1.;
    float beta = 0.;
    auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    float * Aptr = static_cast<float*>(A.data_ptr());
    float * Bptr = static_cast<float*>(B.data_ptr());
    float * Cptr = static_cast<float*>(C.data_ptr());
    int lda = K; // the size with respect stride 1
    int ldb = K; // the size with respect stride 1
    int ldc = N; // the size with respect stride 1
    using CutlassGemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor,
                                                    float, cutlass::layout::ColumnMajor,
                                                    float, cutlass::layout::RowMajor>;
    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K}, {Aptr, lda}, {Bptr, ldb}, {Cptr, ldc}, {Cptr, ldc}, {alpha, beta});
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        puts("CutlassSgemmTN error");
        exit(0);
    }
    return C;
}
