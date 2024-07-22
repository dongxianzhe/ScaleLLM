#include<iostream>
#include<torch/torch.h>


template<int BLOCK_SIZE_N, int BLOCK_SIZE_M>
__global__ void transpose_naive_kernel(float* a, float* o, int M, int N){
    int j = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;
    int i = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
    if(i < M && j < N)o[j * M + i] = a[i * N + j];
}

torch::Tensor transpose_naive(torch::Tensor a){
    const int M = a.size(0);
    const int N = a.size(1);
    torch::Tensor o = torch::empty({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_M = 32;
    dim3 gridDim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    transpose_naive_kernel<BLOCK_SIZE_N, BLOCK_SIZE_M><<<gridDim, blockDim>>>(a.data_ptr<float>(), o.data_ptr<float>(), M, N);
    return o;
}

template<int BLOCK_SIZE_N, int BLOCK_SIZE_M>
__global__ void transpose_smem_kernel(float* a, float* o, int M, int N){
    int j = threadIdx.x + blockIdx.x * BLOCK_SIZE_N;
    int i = threadIdx.y + blockIdx.y * BLOCK_SIZE_M;
    __shared__ float s[32][32];
    if(i < M && j < N){
        s[threadIdx.y][threadIdx.x] = a[i * N + j];
        __syncthreads();
        // int dst_col = threadIdx.x + blockIdx.y * BLOCK_SIZE_M;
        // int dst_row = threadIdx.y + blockIdx.x * BLOCK_SIZE_N;
        // o[dst_col + dst_row * M] = s[threadIdx.x][threadIdx.y];
        o[j * M + i] = s[threadIdx.y][threadIdx.x];
    }
}

torch::Tensor transpose_smem(torch::Tensor a){
    const int M = a.size(0);
    const int N = a.size(1);
    torch::Tensor o = torch::empty({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_M = 32;
    dim3 gridDim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    transpose_smem_kernel<BLOCK_SIZE_N, BLOCK_SIZE_M><<<gridDim, blockDim>>>(a.data_ptr<float>(), o.data_ptr<float>(), M, N);
    return o;
}

template<int BLOCK_SIZE_N, int BLOCK_SIZE_M>
__global__ void transpose_smem_optimized_kernel(float* a, float* o, int M, int N){
    int j = threadIdx.x + blockIdx.x * BLOCK_SIZE_N;
    int i = threadIdx.y + blockIdx.y * BLOCK_SIZE_M;
    __shared__ float s[32][32];
    if(i < M && j < N){
        s[threadIdx.y][threadIdx.x] = a[i * N + j];
        __syncthreads();
        int dst_col = threadIdx.x + blockIdx.y * BLOCK_SIZE_M;
        int dst_row = threadIdx.y + blockIdx.x * BLOCK_SIZE_N;
        o[dst_col + dst_row * M] = s[threadIdx.x][threadIdx.y];
    }
}

torch::Tensor transpose_smem_optimized(torch::Tensor a){
    const int M = a.size(0);
    const int N = a.size(1);
    torch::Tensor o = torch::empty({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_M = 32;
    dim3 gridDim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    transpose_smem_optimized_kernel<BLOCK_SIZE_N, BLOCK_SIZE_M><<<gridDim, blockDim>>>(a.data_ptr<float>(), o.data_ptr<float>(), M, N);
    return o;
}

template<int BLOCK_SIZE_N, int BLOCK_SIZE_M>
__global__ void transpose_smem_optimized_nobankconflict_kernel(float* a, float* o, int M, int N){
    int j = threadIdx.x + blockIdx.x * BLOCK_SIZE_N;
    int i = threadIdx.y + blockIdx.y * BLOCK_SIZE_M;
    __shared__ float s[32][33];
    if(i < M && j < N){
        s[threadIdx.y][threadIdx.x] = a[i * N + j];
        __syncthreads();
        int dst_col = threadIdx.x + blockIdx.y * BLOCK_SIZE_M;
        int dst_row = threadIdx.y + blockIdx.x * BLOCK_SIZE_N;
        o[dst_col + dst_row * M] = s[threadIdx.x][threadIdx.y];
    }
}

torch::Tensor transpose_smem_optimized_nobankconflict(torch::Tensor a){
    const int M = a.size(0);
    const int N = a.size(1);
    torch::Tensor o = torch::empty({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_M = 32;
    dim3 gridDim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    transpose_smem_optimized_nobankconflict_kernel<BLOCK_SIZE_N, BLOCK_SIZE_M><<<gridDim, blockDim>>>(a.data_ptr<float>(), o.data_ptr<float>(), M, N);
    return o;
}

void test(torch::Tensor o, torch::Tensor o_ref, const char* name){
    if(o.is_cuda())o = o.to(torch::kCPU);
    if(o_ref.is_cuda())o_ref = o_ref.to(torch::kCPU);
    const float atol = 1e-3;
    const float rtol = 1e-3;
    if(torch::allclose(o, o_ref, atol, rtol))printf("%s pass\n", name);
    else printf("%s fail\n", name);
}

int main(int argc, char* argv[]){
    if(argc < 3){
        printf("usage: [M] [N]\n");
        return 0;
    }
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    torch::Tensor a = torch::randn({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    torch::Tensor o1 = transpose_naive(a);
    torch::Tensor o2 = transpose_smem(a);
    torch::Tensor o3 = transpose_smem_optimized(a);
    torch::Tensor o4 = transpose_smem_optimized_nobankconflict(a);

    torch::Tensor o_ref = a.transpose(0, 1).contiguous();


    test(o1, o_ref, "transpose_naive");
    test(o2, o_ref, "transpose_smem");
    test(o3, o_ref, "transpose_smem_optimized");
    test(o4, o_ref, "transpose_smem_optimized_nobankconflict");

    // std::cout << a.slice(0, 0, 4).slice(1, 0, 4) << std::endl;
    // std::cout << o.slice(0, 0, 4).slice(1, 0, 4) << std::endl;
    // std::cout << o_ref.slice(0, 0, 4).slice(1, 0, 4) << std::endl;

    return 0;
}