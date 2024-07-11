#include<iostream>
#include<torch/torch.h>
#include"sum.cuh"


void test(torch::Tensor a, torch::Tensor b, std::string name){
    if (a.is_cuda())a = a.to(torch::kCPU);
    if (b.is_cuda())b = b.to(torch::kCPU);
    float eps = 1e-3;
    if (a.allclose(b, eps, eps)) {
        std::cout << name << ": pass" << std::endl;
    } else {
        std::cout << name << ": fail" << std::endl;
    }
}

int main() {
    torch::Tensor a = torch::randn({256 * 256}, torch::device(torch::kCUDA));

    torch::Tensor b1 = sum_naive(a);
    torch::Tensor b2 = sum_nodiverge(a);

    torch::Tensor b_ref = a.sum();

    test(b1, b_ref, "sum_naive");
    test(b2, b_ref, "sum_nodiverge");

    return 0;
}