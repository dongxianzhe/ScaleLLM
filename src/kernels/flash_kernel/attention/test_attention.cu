#include<iostream>
#include<torch/torch.h>
#include"prefill.cuh"

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

int main(){
    int num_qo_head = 32;
    int head_dim = 256;
    int seq_len = 512;
    int num_kv_head = 4;

    auto q = torch::randn({seq_len, num_qo_head, head_dim}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto k = torch::randn({seq_len, num_kv_head, head_dim}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto v = torch::randn({seq_len, num_kv_head, head_dim}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    return 0;
}