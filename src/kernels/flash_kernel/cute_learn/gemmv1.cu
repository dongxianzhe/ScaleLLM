#include<iostream>
#include<torch/torch.h>
#include<cute/tensor.hpp>

int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 256;
    auto a = torch::arange(M * K)
}