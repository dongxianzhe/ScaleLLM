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

int main(int argc, char* argv[]) {
    if(argc != 2){
        fprintf(stderr, "usage: [M]");
        return 0;
    }
    int M = atoi(argv[1]);
    torch::Tensor a = torch::randn({M}, torch::device(torch::kCUDA));

    torch::Tensor b1 = sum_naive(a);
    torch::Tensor b2 = sum_nodiverge(a);
    torch::Tensor b3 = sum_nodiverge_nobankconflict(a);
    torch::Tensor b4 = sum_nodiverge_nobankconflict_nofree(a);
    torch::Tensor b5 = sum_nodiverge_nobankconflict_nofree_lesssync(a);
    torch::Tensor b6 = sum_shuffle(a);

    torch::Tensor b_ref = a.sum();

    test(b1, b_ref, "sum_naive");
    test(b2, b_ref, "sum_nodiverge");
    test(b3, b_ref, "sum_nodiverge_nobankconflict");
    test(b4, b_ref, "sum_nodiverge_nobankconflict_nofree");
    test(b5, b_ref, "sum_nodiverge_nobankconflict_nofree_lesssync");
    test(b6, b_ref, "sum_shuffle");

    return 0;
}