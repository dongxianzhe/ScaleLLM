#include"dispatch.cuh"
#include<torch/torch.h>

__global__ void add_kernel(float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)c[i] = a[i] + b[i];
}

void add(torch::Tensor a, torch::Tensor b, torch::Tensor c){
    const int BLOCK_SIZE = 1024;
    dim3 blockDim(1024);
    dim3 gridDim((a.numel() + 1024 - 1) / BLOCK_SIZE);

    
    add_kernel<<<gridDim, blockDim>>>(
        static_cast<float*>(a.data_ptr()),
        static_cast<float*>(b.data_ptr()),
        static_cast<float*>(c.data_ptr()),
        a.numel());
    
}