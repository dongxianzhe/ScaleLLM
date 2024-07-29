#include"sgemm.cuh"
#include<torch/torch.h>
#include"gemm_cute.cuh"

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

int main(int argc, char* argv[]) {
    if(argc != 4){
        printf("usage: [M] [K] [N]\n");
        exit(0);
    }
    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);

    auto A = torch::randn({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto B = torch::randn({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto A_half = A.to(torch::kHalf);
    auto B_half = B.to(torch::kHalf);


    auto C1 = sgemm_naive(A, B);
    auto C2 = sgemm_smem(A, B);
    auto C3 = sgemm_smem_reg(A, B);
    auto C4 = sgemm_smem_reg_coalesce(A, B);
    auto C5 = sgemm_smem_reg_coalesce_pg2s(A, B);
    auto C6 = sgemm_smem_reg_coalesce_pg2s_ps2r(A, B);
    auto C7 = cute_sgemm_naive(A, B);
    auto C8 = cute_mma_gemm_simple(A_half, B_half);
    auto C9 = cute_gemm_multi_stage(A_half, B_half.transpose(0, 1).contiguous());
    auto C10 = cute_gemm_multi_stage_demo(A_half, B_half);

    auto C_ref = torch::matmul(A, B);
    auto C_ref_half = torch::matmul(A_half, B_half);

    test(C_ref, C1, "sgemm_naive");
    test(C_ref, C2, "sgemm_smem");
    test(C_ref, C3, "sgemm_smem_reg");
    test(C_ref, C4, "sgemm_smem_reg_coalesce");
    test(C_ref, C5, "sgemm_smem_reg_coalesce_pg2s");
    test(C_ref, C6, "sgemm_smem_reg_coalesce_pg2s_ps2r");
    test(C_ref, C7, "cute_sgemm_naive");
    test(C_ref_half, C8, "cute_mma_gemm_simple");
    test(C_ref_half, C9, "cute_gemm_multi_stage");
    test(C_ref_half, C10, "cute_gemm_multi_stage_demo");

    return 0;
}