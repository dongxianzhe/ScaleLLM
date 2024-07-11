#include<torch/torch.h>
#include"common.cuh"

template<int BLOCK_SIZE>
__global__ void sum_naive_kernel(float* in, float* out, int M){
    __shared__ float sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(x < M)sum[tid] = in[x];
    __syncthreads();
    for(int s = 1;s < BLOCK_SIZE; s <<= 1){ // 每个线程控制一个sum[tid]
        if(tid % (2 * s) == 0)sum[tid] += sum[tid + s]; //每次按照2s长度进行分组，第一个线程负责求组内和
        __syncthreads();
    }
    // 最终所有元素属于一个大组，第一个线程知道组内和
    if(tid == 0)out[blockIdx.x] = sum[tid];
}


torch::Tensor sum_naive(torch::Tensor in){
    const int BLOCK_SIZE = 512;
    CHECK(in.dim() == 1, "fsum_naive only support one dimension");
    CHECK(in.size(0) < BLOCK_SIZE * BLOCK_SIZE, "fsum_naive only support elements less than 512 * 512");
    int M = in.size(0);
    int N = div_ceil(M, BLOCK_SIZE);
    auto t = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_naive_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(t.data_ptr()), M);
    auto result = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_naive_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(static_cast<float*>(t.data_ptr()), static_cast<float*>(result.data_ptr()), N);
    return result;
}

template<int BLOCK_SIZE>
__global__ void sum_nodiverge_kernel(float* in, float* out, int M){
    __shared__ float sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    sum[tid] = in[blockIdx.x * BLOCK_SIZE + threadIdx.x]; // 加载数据的时候一个线程负责一个float
    __syncthreads();

    for(int s = 1; s < BLOCK_SIZE; s *= 2){
        // 每次拿出前BLOCK_SIZE / 2 / s个线程进行求和，这样可以防止线程束分化
        int i = 2 * s * tid; // 线程找到自己要进行求和的位置
        if(i < BLOCK_SIZE)sum[i] += sum[i + s];
        __syncthreads();
    }

    if(tid == 0)out[blockIdx.x] = sum[tid];
}

torch::Tensor sum_nodiverge(torch::Tensor in){
    const int BLOCK_SIZE = 512;
    CHECK(in.dim() == 1, "sum_nodiverge only support one dimension");
    CHECK(in.size(0) < BLOCK_SIZE * BLOCK_SIZE, "sum_nodiverge only support elements less than 512 * 512");
    int M = in.size(0);
    int N = div_ceil(M, BLOCK_SIZE);
    auto t = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_nodiverge_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(t.data_ptr()), M);
    auto result = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_nodiverge_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(static_cast<float*>(t.data_ptr()), static_cast<float*>(result.data_ptr()), N);
    return result;
}