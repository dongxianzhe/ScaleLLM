#include<torch/torch.h>

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void sgemm_naive_kernel(float* a, float* b, float* c, int M, int N, int K){
    int i = BLOCK_SIZE_M * blockIdx.x + threadIdx.x;
    int j = BLOCK_SIZE_N * blockIdx.y + threadIdx.y;
    float sum = 0;
    for(int k = 0;k < K;k ++){
        sum += a[i * K + k] * b[k + j * K];
    }
    c[i * N + j] = sum;
}

torch::Tensor sgemm_naive(torch::Tensor a, torch::Tensor b){
    const int M = a.size(0);
    const int N = b.size(0);
    const int K = a.size(1);
    const int BLOCK_SIZE_M = 8;
    const int BLOCK_SIZE_N = 8;

    torch::Tensor c = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 gridDim(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
    dim3 blockDim(BLOCK_SIZE_M, BLOCK_SIZE_N);
    sgemm_naive_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N><<<gridDim, blockDim>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    return c;
}