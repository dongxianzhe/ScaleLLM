#include<iostream>
#include<torch/torch.h>
#include"add.cuh"

int main() {
    torch::Tensor a = torch::randn({10}, torch::device(torch::kCUDA));
    torch::Tensor b = torch::randn({10}, torch::device(torch::kCUDA));
    torch::Tensor c = torch::zeros({10}, torch::device(torch::kCUDA));

    std::cout << "Tensor a:" << std::endl << a << std::endl;
    std::cout << "Tensor b:" << std::endl << b << std::endl;

    add(a, b, c);

    std::cout << "Tensor c (a + b):" << std::endl << c << std::endl;

    auto c_cpu = c.cpu();
    auto a_cpu = a.cpu();
    auto b_cpu = b.cpu();
    auto expected = a_cpu + b_cpu;
    bool passed = torch::allclose(c_cpu, expected);

    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
        std::cout << "Expected:" << std::endl << expected << std::endl;
    }

    return 0;
}