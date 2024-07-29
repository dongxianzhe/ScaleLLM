#include<iostream>
#include<torch/torch.h>
#include"sgemm_naive.cuh"
#include"hgemm.cuh"

void test(torch::Tensor o, torch::Tensor o_ref,const char* name){
    if(o.is_cuda())o = o.to(torch::kCPU);
    if(o_ref.is_cuda())o_ref = o_ref.to(torch::kCPU);

    const float atol = 1e-1;
    const float rtol = 1e-1;

    if(torch::allclose(o, o_ref, atol, rtol))printf("%s pass\n", name);
    else printf("%s fail\n", name);
}

int main(int argc, char* argv[]){
    if(argc < 4){
        printf("usage: [M] [N] [K]\n");
        return 0;
    }
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);

    torch::Tensor a_float32 = torch::randn({M, K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor b_float32 = torch::randn({N, K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor a_half = a_float32.to(torch::kHalf);
    torch::Tensor b_half = b_float32.to(torch::kHalf);

    torch::Tensor c0 = sgemm_naive(a_float32, b_float32);
    torch::Tensor c1 = gemm_asynccp_ldmatrix_tensorcore_cute(a_half, b_half);

    torch::Tensor c_ref_float32 = torch::matmul(a_float32, b_float32.transpose(0, 1));
    torch::Tensor c_ref_half = torch::matmul(a_half, b_half.transpose(0, 1));

    test(c0, c_ref_float32, "sgemm_navie");
    test(c1, c_ref_half, "gemm_asynccp_ldmatrix_tensorcore_cute");

    // std::cout << c0.slice(0, 0, 4).slice(1, 0, 4) << std::endl;
    // std::cout << c_ref_float32.slice(0, 0, 4).slice(1, 0, 4) << std::endl;

    return 0;
}