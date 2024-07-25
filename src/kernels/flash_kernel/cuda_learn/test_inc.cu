#include<iostream>
#include<torch/torch.h>

__global__ void kernel(float*a){
    a[0] ++;
}

int main(){
    torch::Tensor a = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    kernel<<<1, 32 * 8>>>(static_cast<float*>(a.data_ptr()));
    std::cout << a << std::endl;
    kernel<<<1, 32 * 8>>>(static_cast<float*>(a.data_ptr()));
    std::cout << a << std::endl;
    return 0;
}