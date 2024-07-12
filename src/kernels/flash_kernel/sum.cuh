#include<torch/torch.h>
#include"common.cuh"

template<int BLOCK_SIZE>
__global__ void sum_naive_kernel(float* in, float* out, int M){
    __shared__ float sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(x < M)sum[tid] = in[x];
    else sum[tid] = 0;
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
    CHECK_FATAL(in.dim() == 1, "sum_naive only support one dimension");
    CHECK_FATAL(in.size(0) <= BLOCK_SIZE * BLOCK_SIZE, "sum_naive only support elements less than 512 * 512");
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
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    // 加载数据的时候一个线程负责一个float
    if(x < M)sum[tid] = in[x];
    else sum[tid] = 0;
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
    CHECK_FATAL(in.dim() == 1, "sum_nodiverge only support one dimension");
    CHECK_FATAL(in.size(0) <= BLOCK_SIZE * BLOCK_SIZE, "sum_nodiverge only support elements less than 512 * 512");
    int M = in.size(0);
    int N = div_ceil(M, BLOCK_SIZE);
    auto t = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_nodiverge_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(t.data_ptr()), M);
    auto result = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_nodiverge_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(static_cast<float*>(t.data_ptr()), static_cast<float*>(result.data_ptr()), N);
    return result;
}

template<int BLOCK_SIZE>
__global__ void sum_nodiverge_nobankconflict_kernel(float* in, float* out, int M){
    __shared__ float sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(x < M)sum[tid] = in[x];
    else sum[tid] = 0;
    __syncthreads();
    for(int s = BLOCK_SIZE / 2;s > 0;s >>= 1){
        if(tid < s)sum[tid] += sum[tid + s];
        __syncthreads();
    }
    if(tid == 0)out[blockIdx.x] = sum[tid];
}

torch::Tensor sum_nodiverge_nobankconflict(torch::Tensor in){
    const int BLOCK_SIZE = 512;
    CHECK_FATAL(in.dim() == 1, "sum_nodiverge only support one dimension");
    CHECK_FATAL(in.size(0) <= BLOCK_SIZE * BLOCK_SIZE, "sum_nodiverge only support elements less than 512 * 512");
    int M = in.size(0);
    int N = div_ceil(M, BLOCK_SIZE);
    torch::Tensor out;
    while(N != 1){
        out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
        sum_nodiverge_nobankconflict_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
        in = out;
        M = N;
        N = div_ceil(M, BLOCK_SIZE);
    }
    out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_nodiverge_nobankconflict_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
    return out;
}

template<int BLOCK_SIZE>
__global__ void sum_nodiverge_nobankconflict_nofree_kernel(float* in, float* out, int M){
    __shared__ float sum[BLOCK_SIZE / 2];
    int tid = threadIdx.x;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = x + BLOCK_SIZE / 2;
    if(y < M)sum[tid] = in[x] + in[y];
    else if(x < M)sum[tid] = in[x];
    else sum[tid] = 0;
    __syncthreads();
    for(int s = BLOCK_SIZE / 4; s > 0;s >>= 1){
        if(tid < s)sum[tid] += sum[tid + s];
        __syncthreads();
    }
    if(tid == 0)out[blockIdx.x] = sum[tid];
}

torch::Tensor sum_nodiverge_nobankconflict_nofree(torch::Tensor in){
    const int BLOCK_SIZE = 512;
    CHECK_FATAL(in.dim() == 1, "sum_nodiverge only support one dimension");
    CHECK_FATAL(in.size(0) <= BLOCK_SIZE * BLOCK_SIZE, "sum_nodiverge only support elements less than 512 * 512");
    int M = in.size(0);
    int N = div_ceil(M, BLOCK_SIZE);
    torch::Tensor out;
    while(N != 1){
        out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
        sum_nodiverge_nobankconflict_nofree_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE / 2>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
        in = out;
        M = N;
        N = div_ceil(M, BLOCK_SIZE);
    }
    out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_nodiverge_nobankconflict_nofree_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE / 2>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
    return out;
}

template<int BLOCK_SIZE>
__global__ void sum_nodiverge_nobankconflict_nofree_lesssync_kernel(float* in, float* out, int M){
    __shared__ float sum[BLOCK_SIZE / 2];
    int tid = threadIdx.x;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = x + BLOCK_SIZE / 2;
    if(y < M)sum[tid] = in[x] + in[y];
    else if(x < M)sum[tid] = in[x];
    else sum[tid] = 0;
    __syncthreads();
    for(int s = BLOCK_SIZE / 4; s > 32;s >>= 1){
        if(tid < s)sum[tid] += sum[tid + s];
        __syncthreads();
    }
    if(tid < 32){
        volatile float* t = sum; // volatile指针可以防止编译期对取数据进行优化
        t[tid] += t[tid + 32];
        t[tid] += t[tid + 16];
        t[tid] += t[tid + 8];
        t[tid] += t[tid + 4];
        t[tid] += t[tid + 2];
        t[tid] += t[tid + 1];
    }
    if(tid == 0)out[blockIdx.x] = sum[tid];
}

torch::Tensor sum_nodiverge_nobankconflict_nofree_lesssync(torch::Tensor in){
    const int BLOCK_SIZE = 512;
    CHECK_FATAL(in.dim() == 1, "sum_nodiverge only support one dimension");
    CHECK_FATAL(in.size(0) <= BLOCK_SIZE * BLOCK_SIZE, "sum_nodiverge only support elements less than 512 * 512");
    int M = in.size(0);
    int N = div_ceil(M, BLOCK_SIZE);
    torch::Tensor out;
    while(N != 1){
        out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
        sum_nodiverge_nobankconflict_nofree_lesssync_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE / 2>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
        in = out;
        M = N;
        N = div_ceil(M, BLOCK_SIZE);
    }
    out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_nodiverge_nobankconflict_nofree_lesssync_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE / 2>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
    return out;
}

template<int NUM_THREADS>
__device__ __forceinline__ float shuffle_warp_reduce_sum(float sum){
    if(NUM_THREADS >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16);
    if(NUM_THREADS >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);
    if(NUM_THREADS >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);
    if(NUM_THREADS >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);
    if(NUM_THREADS >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template<int BLOCK_SIZE, int NUM_THREADS>
__global__ void sum_shuffle_kernel(float* in, float* out, int M){
    float sum = 0;
    int tid = threadIdx.x;
    #pragma unroll
    for(int i = 0;i < BLOCK_SIZE;i += NUM_THREADS){
        int x = blockIdx.x * BLOCK_SIZE + threadIdx.x + i;
        if(x < M)sum += in[x];
    }

    sum = shuffle_warp_reduce_sum<NUM_THREADS>(sum);
    const int warp_size = 32;
    const int num_warps = NUM_THREADS / warp_size;
    __shared__ float warp_sum[num_warps];
    int warp_id = tid / warp_size, lane_id = tid % warp_size;
    if(lane_id == 0)warp_sum[warp_id] = sum;
    __syncthreads();

    if(tid < num_warps)sum = warp_sum[tid];
    else sum = 0;
    if(warp_id == 0)sum = shuffle_warp_reduce_sum<num_warps>(sum);
    if(tid == 0)out[blockIdx.x] = sum;
}


torch::Tensor sum_shuffle(torch::Tensor in){
    const int BLOCK_SIZE = 512;
    const int NUM_THREADS = 128;
    CHECK_FATAL(in.dim() == 1, "sum_nodiverge only support one dimension");
    CHECK_FATAL(in.size(0) <= BLOCK_SIZE * BLOCK_SIZE, "sum_nodiverge only support elements less than 512 * 512");

    int M = in.size(0);
    int N = div_ceil(M, BLOCK_SIZE);
    torch::Tensor out;
    while(N != 1){
        out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
        sum_shuffle_kernel<BLOCK_SIZE, NUM_THREADS><<<N, NUM_THREADS>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
        in = out;
        M = N;
        N = div_ceil(M, BLOCK_SIZE);
    }
    out = torch::zeros({N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    sum_shuffle_kernel<BLOCK_SIZE, NUM_THREADS><<<N, NUM_THREADS>>>(static_cast<float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), M);
    return out;
}