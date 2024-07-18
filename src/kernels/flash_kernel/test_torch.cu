#include<iostream>
#include<torch/torch.h>

int main(){
    const int n = 10;
    auto a = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = a.to(torch::kHalf);
    auto c = b.to(torch::kHalf);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    bool sus = torch::allclose(c, b, 1e-3);
    if(sus)puts("yes");
    else puts("no");
    return 0;
}