#include<torch/torch.h>
#include"dispatch.cuh"
#include"common.cuh"
#include<cassert>

template<int BLOCK_SIZE_M, int BLOCK_SIZE_N>
__global__ void sgemm_naive_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int K, int N){
    int i = blockIdx.x * BLOCK_SIZE_M + threadIdx.x;
    int j = blockIdx.y * BLOCK_SIZE_N + threadIdx.y;
    if(i >= M || j >= N)return;
    float sum = 0;
    for(int k = 0;k < K;k ++){
        sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

torch::Tensor sgemm_naive(torch::Tensor A, torch::Tensor B){
    CHECK(A.dim() == 2, "A matrix dim should be 2");
    CHECK(B.dim() == 2, "B matrix dim should be 2");
    CHECK(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK(A.scalar_type() == at::ScalarType::Float, "gemm naive only support float");
    CHECK(B.scalar_type() == at::ScalarType::Float, "gemm naive only support float");
    CHECK(A.scalar_type() == B.scalar_type(), "gemm matrix A and B should have same type");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(A.scalar_type(), c_type, [&]{
        const int BLOCK_SIZE_M = 8;
        const int BLOCK_SIZE_N = 8;
        const int NUM_BLOCK_M = div_ceil(M, BLOCK_SIZE_M);
        const int NUM_BLOCK_N = div_ceil(N, BLOCK_SIZE_N);
        dim3 gridDim(NUM_BLOCK_M, NUM_BLOCK_N);
        dim3 blockDim(BLOCK_SIZE_M, BLOCK_SIZE_N);
        sgemm_naive_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N><<<gridDim, blockDim>>>(
            static_cast<float*>(A.data_ptr()), 
            static_cast<float*>(B.data_ptr()), 
            static_cast<float*>(C.data_ptr()),
            M, K, N
        );
        return true;
    });
    return C;
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_K, int BLOCK_SIZE_N>
__global__ void sgemm_smem_kernel(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    const int NUM_BLOCK_K = K / BLOCK_SIZE_K;
    const int bm = blockIdx.x, bn = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    

    float sum = 0;
    for(int bk = 0;bk < NUM_BLOCK_K;bk ++){
        As[ty][tx] = A[OFFSET(bm * BLOCK_SIZE_M + tx, bk * BLOCK_SIZE_K + ty, K)];
        Bs[ty][tx] = B[OFFSET(bk * BLOCK_SIZE_K + ty, bn * BLOCK_SIZE_N + tx, N)];
        __syncthreads();
        for(int k = 0;k < BLOCK_SIZE_K;k ++){
            sum += As[k][tx] * Bs[k][ty]; 
        }
        __syncthreads();
    }
    C[OFFSET(bm * BLOCK_SIZE_M + tx, bn * BLOCK_SIZE_N + ty, N)] = sum;
}

torch::Tensor sgemm_smem(torch::Tensor A, torch::Tensor B){
    CHECK(A.dim() == 2, "A matrix dim should be 2");
    CHECK(B.dim() == 2, "B matrix dim should be 2");
    CHECK(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK(A.size(0) % 8 == 0, "gemm smem only support M mod 8 == 0")
    CHECK(A.size(1) % 8 == 0, "gemm smem only support K mod 8 == 0")
    CHECK(B.size(1) % 8 == 0, "gemm smem only support N mod 8 == 0")
    CHECK(A.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(B.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(A.scalar_type() == B.scalar_type(), "gemm smem matrix A and B should have same type");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(A.scalar_type(), c_type, [&]{
        const int BLOCK_SIZE_K = 8;
        const int BLOCK_SIZE_M = BLOCK_SIZE_K;
        const int BLOCK_SIZE_N = BLOCK_SIZE_K;
        const int NUM_BLOCK_M = div_ceil(M, BLOCK_SIZE_M);
        const int NUM_BLOCK_N = div_ceil(N, BLOCK_SIZE_N);
        dim3 gridDim(NUM_BLOCK_M, NUM_BLOCK_N);
        dim3 blockDim(BLOCK_SIZE_M, BLOCK_SIZE_N);
        sgemm_smem_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>(
            static_cast<float*>(A.data_ptr()), 
            static_cast<float*>(B.data_ptr()), 
            static_cast<float*>(C.data_ptr()),
            M, K, N
        );
        return true;
    });
    return C;
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_K, int BLOCK_SIZE_N, int TILE_SIZE_M, int TILE_SIZE_N>
__global__ void sgemm_smem_reg_kernel(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    int bm = blockIdx.x, bn = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * blockDim.x + tx;
    const int NUM_THREADS_M = BLOCK_SIZE_M / TILE_SIZE_M;
    const int NUM_THREADS_N = BLOCK_SIZE_N / TILE_SIZE_N;
    const int NUM_THREADS = NUM_THREADS_M * NUM_THREADS_N;

    const int num_float_per_fetch = 128 / 8 / sizeof(float);
    const int num_ldg_block_a = BLOCK_SIZE_M * BLOCK_SIZE_K / NUM_THREADS / num_float_per_fetch;
    const int num_ldg_block_b = BLOCK_SIZE_K * BLOCK_SIZE_N / NUM_THREADS / num_float_per_fetch;
    const int ldg_block_a_stride = BLOCK_SIZE_M / num_ldg_block_a;
    const int ldg_block_b_stride = BLOCK_SIZE_K / num_ldg_block_b;

    float sum[TILE_SIZE_M][TILE_SIZE_N] = {0};
    const int NUM_BLOCK_K = K / BLOCK_SIZE_K;
    for(int bk = 0;bk < NUM_BLOCK_K;bk ++){
        // load block a
        for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
            const int ldg_block_a_threads_per_row = BLOCK_SIZE_K / num_float_per_fetch;
            for(int j = 0;j < num_float_per_fetch;j ++){
                As[tid % ldg_block_a_threads_per_row * num_float_per_fetch + j][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = A[OFFSET(
                    bm * BLOCK_SIZE_M + ldg_a_offset_m + tid / ldg_block_a_threads_per_row,
                    bk * BLOCK_SIZE_K + tid % ldg_block_a_threads_per_row * num_float_per_fetch + j,
                    K)];
            }
        }
        // load block b
        for(int ldg_b_offset_k = 0;ldg_b_offset_k < BLOCK_SIZE_K;ldg_b_offset_k += ldg_block_b_stride){
            const int ldg_block_b_threads_per_row = BLOCK_SIZE_N / num_float_per_fetch;
            for(int j = 0;j < num_float_per_fetch;j ++){
                Bs[ldg_b_offset_k + tid / ldg_block_b_threads_per_row][tid % ldg_block_b_threads_per_row * num_float_per_fetch + j] = B[OFFSET(
                    bk * BLOCK_SIZE_K + ldg_b_offset_k + tid / ldg_block_b_threads_per_row,
                    bn * BLOCK_SIZE_N + tid % ldg_block_b_threads_per_row * num_float_per_fetch + j,
                    N)];
            }
        }
        __syncthreads();
        float frag_a[TILE_SIZE_M];
        float frag_b[TILE_SIZE_N];
        for(int k = 0;k < BLOCK_SIZE_K;k ++){
            // load frag a
            for(int frag_a_offset_m = 0; frag_a_offset_m < BLOCK_SIZE_M; frag_a_offset_m += num_float_per_fetch){
                for(int j = 0;j < num_float_per_fetch;j ++){
                    frag_a[frag_a_offset_m + j] = As[k][tx * TILE_SIZE_M + frag_a_offset_m + j];
                }
            }
            // load frag b
            for(int frag_b_offset_n = 0; frag_b_offset_n < BLOCK_SIZE_N; frag_b_offset_n += num_float_per_fetch){
                for(int j = 0;j < num_float_per_fetch;j ++){
                    frag_b[frag_b_offset_n + j] = Bs[k][ty * TILE_SIZE_N + frag_b_offset_n + j];
                }
            }
            // compute
            for(int i = 0;i < TILE_SIZE_M;i ++){
                for(int j = 0;j < TILE_SIZE_N;j ++){
                    sum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }
        __syncthreads();
    }
    // store tile c
    for(int i = 0;i < TILE_SIZE_M;i ++){
        for(int j = 0;j < TILE_SIZE_N;j ++){
            C[OFFSET(bm * BLOCK_SIZE_M + tx * TILE_SIZE_M + i, bn * BLOCK_SIZE_N + ty * TILE_SIZE_N + j, N)] = sum[i][j];
        }
    }
}

torch::Tensor sgemm_smem_reg(torch::Tensor A, torch::Tensor B){
    CHECK(A.dim() == 2, "A matrix dim should be 2");
    CHECK(B.dim() == 2, "B matrix dim should be 2");
    CHECK(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK(A.size(0) % 128 == 0, "gemm smem only support M mod 128 == 0")
    CHECK(A.size(1) % 8 == 0, "gemm smem only support K mod 8 == 0")
    CHECK(B.size(1) % 128 == 0, "gemm smem only support N mod 128 == 0")
    CHECK(A.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(B.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(A.scalar_type() == B.scalar_type(), "gemm smem matrix A and B should have same type");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(A.scalar_type(), c_type, [&]{
        const int BLOCK_SIZE_M = 128;
        const int BLOCK_SIZE_K = 8;
        const int BLOCK_SIZE_N = 128;
        const int NUM_BLOCK_M = M / BLOCK_SIZE_M;
        const int NUM_BLOCK_N = N / BLOCK_SIZE_N;

        const int TILE_SIZE_M = 8;
        const int TILE_SIZE_N = 8;
        const int NUM_TILE_M = BLOCK_SIZE_M / TILE_SIZE_M;
        const int NUM_SIZE_N = BLOCK_SIZE_N / TILE_SIZE_N;
        dim3 gridDim(NUM_BLOCK_M, NUM_BLOCK_N);
        dim3 blockDim(NUM_TILE_M, NUM_SIZE_N);
        sgemm_smem_reg_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, TILE_SIZE_M, TILE_SIZE_N><<<gridDim, blockDim>>>(
            static_cast<float*>(A.data_ptr()), 
            static_cast<float*>(B.data_ptr()), 
            static_cast<float*>(C.data_ptr()),
            M, K, N);
        return true;
    });
    return C;
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_K, int BLOCK_SIZE_N, int TILE_SIZE_M, int TILE_SIZE_N>
__global__ void sgemm_smem_reg_coalesce_kernel(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    int bm = blockIdx.x, bn = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * blockDim.x + tx;
    const int NUM_THREADS_M = BLOCK_SIZE_M / TILE_SIZE_M;
    const int NUM_THREADS_N = BLOCK_SIZE_N / TILE_SIZE_N;
    const int NUM_THREADS = NUM_THREADS_M * NUM_THREADS_N;

    const int num_float_per_fetch = 128 / 8 / sizeof(float);
    const int num_ldg_block_a = BLOCK_SIZE_M * BLOCK_SIZE_K / NUM_THREADS / num_float_per_fetch;
    const int num_ldg_block_b = BLOCK_SIZE_K * BLOCK_SIZE_N / NUM_THREADS / num_float_per_fetch;
    const int ldg_block_a_stride = BLOCK_SIZE_M / num_ldg_block_a;
    const int ldg_block_b_stride = BLOCK_SIZE_K / num_ldg_block_b;

    float sum[TILE_SIZE_M][TILE_SIZE_N] = {0};
    const int NUM_BLOCK_K = K / BLOCK_SIZE_K;
    for(int bk = 0;bk < NUM_BLOCK_K;bk ++){
        // load block a
        for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
            const int ldg_block_a_threads_per_row = BLOCK_SIZE_K / num_float_per_fetch;
            float ldg_a_buffer[num_float_per_fetch];
            FLOAT4(ldg_a_buffer[0]) = 
            FLOAT4(A[OFFSET(
                bm * BLOCK_SIZE_M + ldg_a_offset_m + tid / ldg_block_a_threads_per_row,
                bk * BLOCK_SIZE_K + tid % ldg_block_a_threads_per_row * num_float_per_fetch,
                K)]);
            As[tid % ldg_block_a_threads_per_row * num_float_per_fetch + 0][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[0];
            As[tid % ldg_block_a_threads_per_row * num_float_per_fetch + 1][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[1];
            As[tid % ldg_block_a_threads_per_row * num_float_per_fetch + 2][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[2];
            As[tid % ldg_block_a_threads_per_row * num_float_per_fetch + 3][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[3];
        }
        // load block b
        for(int ldg_b_offset_k = 0;ldg_b_offset_k < BLOCK_SIZE_K;ldg_b_offset_k += ldg_block_b_stride){
            const int ldg_block_b_threads_per_row = BLOCK_SIZE_N / num_float_per_fetch;
            FLOAT4(Bs[ldg_b_offset_k + tid / ldg_block_b_threads_per_row][tid % ldg_block_b_threads_per_row * num_float_per_fetch]) = FLOAT4(B[OFFSET(
                bk * BLOCK_SIZE_K + ldg_b_offset_k + tid / ldg_block_b_threads_per_row,
                bn * BLOCK_SIZE_N + tid % ldg_block_b_threads_per_row * num_float_per_fetch,
                N)]);
        }
        __syncthreads();
        float frag_a[TILE_SIZE_M];
        float frag_b[TILE_SIZE_N];
        for(int k = 0;k < BLOCK_SIZE_K;k ++){
            // load frag a
            for(int frag_a_offset_m = 0; frag_a_offset_m < TILE_SIZE_M; frag_a_offset_m += num_float_per_fetch){
                FLOAT4(frag_a[frag_a_offset_m]) = FLOAT4(As[k][tx * TILE_SIZE_M + frag_a_offset_m]);
            }
            // load frag b
            for(int frag_b_offset_n = 0; frag_b_offset_n < TILE_SIZE_N; frag_b_offset_n += num_float_per_fetch){
                FLOAT4(frag_b[frag_b_offset_n]) = FLOAT4(Bs[k][ty * TILE_SIZE_N + frag_b_offset_n]);
            }
            // compute
            for(int i = 0;i < TILE_SIZE_M;i ++){
                for(int j = 0;j < TILE_SIZE_N;j ++){
                    sum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }
        __syncthreads();
    }
    // store tile c
    for(int i = 0;i < TILE_SIZE_M;i ++){
        for(int j = 0;j < TILE_SIZE_N;j += num_float_per_fetch){
            FLOAT4(C[OFFSET(bm * BLOCK_SIZE_M + tx * TILE_SIZE_M + i, bn * BLOCK_SIZE_N + ty * TILE_SIZE_N + j, N)]) = FLOAT4(sum[i][j]);
        }
    }
}

torch::Tensor sgemm_smem_reg_coalesce(torch::Tensor A, torch::Tensor B){
    CHECK(A.dim() == 2, "A matrix dim should be 2");
    CHECK(B.dim() == 2, "B matrix dim should be 2");
    CHECK(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK(A.size(0) % 128 == 0, "gemm smem only support M mod 128 == 0")
    CHECK(A.size(1) % 8 == 0, "gemm smem only support K mod 8 == 0")
    CHECK(B.size(1) % 128 == 0, "gemm smem only support N mod 128 == 0")
    CHECK(A.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(B.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(A.scalar_type() == B.scalar_type(), "gemm smem matrix A and B should have same type");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(A.scalar_type(), c_type, [&]{
        const int BLOCK_SIZE_M = 128;
        const int BLOCK_SIZE_K = 8;
        const int BLOCK_SIZE_N = 128;
        const int NUM_BLOCK_M = M / BLOCK_SIZE_M;
        const int NUM_BLOCK_N = N / BLOCK_SIZE_N;

        const int TILE_SIZE_M = 8;
        const int TILE_SIZE_N = 8;
        const int NUM_TILE_M = BLOCK_SIZE_M / TILE_SIZE_M;
        const int NUM_SIZE_N = BLOCK_SIZE_N / TILE_SIZE_N;
        dim3 gridDim(NUM_BLOCK_M, NUM_BLOCK_N);
        dim3 blockDim(NUM_TILE_M, NUM_SIZE_N);
        sgemm_smem_reg_coalesce_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, TILE_SIZE_M, TILE_SIZE_N><<<gridDim, blockDim>>>(
            static_cast<float*>(A.data_ptr()), 
            static_cast<float*>(B.data_ptr()), 
            static_cast<float*>(C.data_ptr()),
            M, K, N);
        return true;
    });
    return C;
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_K, int BLOCK_SIZE_N, int TILE_SIZE_M, int TILE_SIZE_N>
__global__ void sgemm_smem_reg_coalesce_pg2s_kernel(float* A, float* B, float* C, int M, int K, int N){
    const int num_pg2s_stages = 2;
    __shared__ float As[num_pg2s_stages][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[num_pg2s_stages][BLOCK_SIZE_K][BLOCK_SIZE_N];
    int bm = blockIdx.x, bn = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * blockDim.x + tx;
    const int NUM_THREADS_M = BLOCK_SIZE_M / TILE_SIZE_M;
    const int NUM_THREADS_N = BLOCK_SIZE_N / TILE_SIZE_N;
    const int NUM_THREADS = NUM_THREADS_M * NUM_THREADS_N;

    const int num_float_per_fetch = 128 / 8 / sizeof(float);
    const int num_ldg_block_a = BLOCK_SIZE_M * BLOCK_SIZE_K / NUM_THREADS / num_float_per_fetch;
    const int num_ldg_block_b = BLOCK_SIZE_K * BLOCK_SIZE_N / NUM_THREADS / num_float_per_fetch;
    const int ldg_block_a_stride = BLOCK_SIZE_M / num_ldg_block_a;
    const int ldg_block_b_stride = BLOCK_SIZE_K / num_ldg_block_b;
    const int ldg_block_a_threads_per_row = BLOCK_SIZE_K / num_float_per_fetch;
    const int ldg_block_b_threads_per_row = BLOCK_SIZE_N / num_float_per_fetch;

    float sum[TILE_SIZE_M][TILE_SIZE_N] = {0};
    const int NUM_BLOCK_K = K / BLOCK_SIZE_K;
    for(int bk = 0;bk < NUM_BLOCK_K;bk ++){
        int pg2s_stage_id = bk & 1;
        int next_pg2s_stage_id = (bk + 1) & 1;
        float ldg_a_buffer[num_float_per_fetch];
        if(bk == 0){
            // load first block a
            for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
                FLOAT4(ldg_a_buffer[0]) = 
                FLOAT4(A[OFFSET(
                    bm * BLOCK_SIZE_M + ldg_a_offset_m + tid / ldg_block_a_threads_per_row,
                    bk * BLOCK_SIZE_K + tid % ldg_block_a_threads_per_row * num_float_per_fetch,
                    K)]);
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 0][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[0];
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 1][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[1];
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 2][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[2];
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 3][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[3];
            }
            // load first block b
            for(int ldg_b_offset_k = 0;ldg_b_offset_k < BLOCK_SIZE_K;ldg_b_offset_k += ldg_block_b_stride){
                FLOAT4(Bs[pg2s_stage_id][ldg_b_offset_k + tid / ldg_block_b_threads_per_row][tid % ldg_block_b_threads_per_row * num_float_per_fetch]) = FLOAT4(B[OFFSET(
                    bk * BLOCK_SIZE_K + ldg_b_offset_k + tid / ldg_block_b_threads_per_row,
                    bn * BLOCK_SIZE_N + tid % ldg_block_b_threads_per_row * num_float_per_fetch,
                    N)]);
            }
            __syncthreads();
        }
        if(bk + 1 < NUM_BLOCK_K){
            // load first block a
            for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
                const int ldg_block_a_threads_per_row = BLOCK_SIZE_K / num_float_per_fetch;
                FLOAT4(ldg_a_buffer[0]) = 
                FLOAT4(A[OFFSET(
                    bm * BLOCK_SIZE_M + ldg_a_offset_m + tid / ldg_block_a_threads_per_row,
                    (bk + 1) * BLOCK_SIZE_K + tid % ldg_block_a_threads_per_row * num_float_per_fetch,
                    K)]);
            }
            // load first block b
            for(int ldg_b_offset_k = 0;ldg_b_offset_k < BLOCK_SIZE_K;ldg_b_offset_k += ldg_block_b_stride){
                const int ldg_block_b_threads_per_row = BLOCK_SIZE_N / num_float_per_fetch;
                FLOAT4(Bs[next_pg2s_stage_id][ldg_b_offset_k + tid / ldg_block_b_threads_per_row][tid % ldg_block_b_threads_per_row * num_float_per_fetch]) = FLOAT4(B[OFFSET(
                    (bk + 1) * BLOCK_SIZE_K + ldg_b_offset_k + tid / ldg_block_b_threads_per_row,
                    bn * BLOCK_SIZE_N + tid % ldg_block_b_threads_per_row * num_float_per_fetch,
                    N)]);
            }
        }
        float frag_a[TILE_SIZE_M];
        float frag_b[TILE_SIZE_N];
        for(int k = 0;k < BLOCK_SIZE_K;k ++){
            // load frag a
            for(int frag_a_offset_m = 0; frag_a_offset_m < TILE_SIZE_M; frag_a_offset_m += num_float_per_fetch){
                FLOAT4(frag_a[frag_a_offset_m]) = FLOAT4(As[pg2s_stage_id][k][tx * TILE_SIZE_M + frag_a_offset_m]);
            }
            // load frag b
            for(int frag_b_offset_n = 0; frag_b_offset_n < TILE_SIZE_N; frag_b_offset_n += num_float_per_fetch){
                FLOAT4(frag_b[frag_b_offset_n]) = FLOAT4(Bs[pg2s_stage_id][k][ty * TILE_SIZE_N + frag_b_offset_n]);
            }
            // compute
            for(int i = 0;i < TILE_SIZE_M;i ++){
                for(int j = 0;j < TILE_SIZE_N;j ++){
                    sum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }

        if(bk + 1 < NUM_BLOCK_K){
            for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 0][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[0];
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 1][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[1];
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 2][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[2];
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 3][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[3];
            }
            __syncthreads();
        }
    }
    // store tile c
    for(int i = 0;i < TILE_SIZE_M;i ++){
        for(int j = 0;j < TILE_SIZE_N;j += num_float_per_fetch){
            FLOAT4(C[OFFSET(bm * BLOCK_SIZE_M + tx * TILE_SIZE_M + i, bn * BLOCK_SIZE_N + ty * TILE_SIZE_N + j, N)]) = FLOAT4(sum[i][j]);
        }
    }
}

torch::Tensor sgemm_smem_reg_coalesce_pg2s(torch::Tensor A, torch::Tensor B){
    CHECK(A.dim() == 2, "A matrix dim should be 2");
    CHECK(B.dim() == 2, "B matrix dim should be 2");
    CHECK(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK(A.size(0) % 128 == 0, "gemm smem only support M mod 128 == 0")
    CHECK(A.size(1) % 8 == 0, "gemm smem only support K mod 8 == 0")
    CHECK(B.size(1) % 128 == 0, "gemm smem only support N mod 128 == 0")
    CHECK(A.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(B.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(A.scalar_type() == B.scalar_type(), "gemm smem matrix A and B should have same type");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(A.scalar_type(), c_type, [&]{
        const int BLOCK_SIZE_M = 128;
        const int BLOCK_SIZE_K = 8;
        const int BLOCK_SIZE_N = 128;
        const int NUM_BLOCK_M = M / BLOCK_SIZE_M;
        const int NUM_BLOCK_N = N / BLOCK_SIZE_N;

        const int TILE_SIZE_M = 8;
        const int TILE_SIZE_N = 8;
        const int NUM_TILE_M = BLOCK_SIZE_M / TILE_SIZE_M;
        const int NUM_SIZE_N = BLOCK_SIZE_N / TILE_SIZE_N;
        dim3 gridDim(NUM_BLOCK_M, NUM_BLOCK_N);
        dim3 blockDim(NUM_TILE_M, NUM_SIZE_N);
        sgemm_smem_reg_coalesce_pg2s_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, TILE_SIZE_M, TILE_SIZE_N><<<gridDim, blockDim>>>(
            static_cast<float*>(A.data_ptr()), 
            static_cast<float*>(B.data_ptr()), 
            static_cast<float*>(C.data_ptr()),
            M, K, N);
        return true;
    });
    return C;
}

template<int BLOCK_SIZE_M, int BLOCK_SIZE_K, int BLOCK_SIZE_N, int TILE_SIZE_M, int TILE_SIZE_N>
__global__ void sgemm_smem_reg_coalesce_pg2s_ps2r_kernel(float* A, float* B, float* C, int M, int K, int N){
    const int num_pg2s_stages = 2;
    __shared__ float As[num_pg2s_stages][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[num_pg2s_stages][BLOCK_SIZE_K][BLOCK_SIZE_N];
    int bm = blockIdx.x, bn = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y, tid = ty * blockDim.x + tx;
    const int NUM_THREADS_M = BLOCK_SIZE_M / TILE_SIZE_M;
    const int NUM_THREADS_N = BLOCK_SIZE_N / TILE_SIZE_N;
    const int NUM_THREADS = NUM_THREADS_M * NUM_THREADS_N;

    const int num_float_per_fetch = 128 / 8 / sizeof(float);
    const int num_ldg_block_a = BLOCK_SIZE_M * BLOCK_SIZE_K / NUM_THREADS / num_float_per_fetch;
    const int num_ldg_block_b = BLOCK_SIZE_K * BLOCK_SIZE_N / NUM_THREADS / num_float_per_fetch;
    const int ldg_block_a_stride = BLOCK_SIZE_M / num_ldg_block_a;
    const int ldg_block_b_stride = BLOCK_SIZE_K / num_ldg_block_b;
    const int ldg_block_a_threads_per_row = BLOCK_SIZE_K / num_float_per_fetch;
    const int ldg_block_b_threads_per_row = BLOCK_SIZE_N / num_float_per_fetch;

    float sum[TILE_SIZE_M][TILE_SIZE_N] = {0};

    const int NUM_BLOCK_K = K / BLOCK_SIZE_K;
    #pragma unroll
    for(int bk = 0;bk < NUM_BLOCK_K;bk ++){
        int pg2s_stage_id = bk & 1;
        int next_pg2s_stage_id = (bk + 1) & 1;
        float ldg_a_buffer[num_float_per_fetch];
        if(bk == 0){
            // load first block a
            #pragma unroll
            for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
                FLOAT4(ldg_a_buffer[0]) = 
                FLOAT4(A[OFFSET(
                    bm * BLOCK_SIZE_M + ldg_a_offset_m + tid / ldg_block_a_threads_per_row,
                    bk * BLOCK_SIZE_K + tid % ldg_block_a_threads_per_row * num_float_per_fetch,
                    K)]);
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 0][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[0];
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 1][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[1];
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 2][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[2];
                As[pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 3][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[3];
            }
            // load first block b
            #pragma unroll
            for(int ldg_b_offset_k = 0;ldg_b_offset_k < BLOCK_SIZE_K;ldg_b_offset_k += ldg_block_b_stride){
                FLOAT4(Bs[pg2s_stage_id][ldg_b_offset_k + tid / ldg_block_b_threads_per_row][tid % ldg_block_b_threads_per_row * num_float_per_fetch]) = FLOAT4(B[OFFSET(
                    bk * BLOCK_SIZE_K + ldg_b_offset_k + tid / ldg_block_b_threads_per_row,
                    bn * BLOCK_SIZE_N + tid % ldg_block_b_threads_per_row * num_float_per_fetch,
                    N)]);
            }
            __syncthreads();
        }
        if(bk + 1 < NUM_BLOCK_K){
            // load first block a
            #pragma unroll
            for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
                const int ldg_block_a_threads_per_row = BLOCK_SIZE_K / num_float_per_fetch;
                FLOAT4(ldg_a_buffer[0]) = 
                FLOAT4(A[OFFSET(
                    bm * BLOCK_SIZE_M + ldg_a_offset_m + tid / ldg_block_a_threads_per_row,
                    (bk + 1) * BLOCK_SIZE_K + tid % ldg_block_a_threads_per_row * num_float_per_fetch,
                    K)]);
            }
            // load first block b
            #pragma unroll
            for(int ldg_b_offset_k = 0;ldg_b_offset_k < BLOCK_SIZE_K;ldg_b_offset_k += ldg_block_b_stride){
                const int ldg_block_b_threads_per_row = BLOCK_SIZE_N / num_float_per_fetch;
                FLOAT4(Bs[next_pg2s_stage_id][ldg_b_offset_k + tid / ldg_block_b_threads_per_row][tid % ldg_block_b_threads_per_row * num_float_per_fetch]) = FLOAT4(B[OFFSET(
                    (bk + 1) * BLOCK_SIZE_K + ldg_b_offset_k + tid / ldg_block_b_threads_per_row,
                    bn * BLOCK_SIZE_N + tid % ldg_block_b_threads_per_row * num_float_per_fetch,
                    N)]);
            }
        }

        const int num_ps2r_stages = 2;
        float frag_a[num_ps2r_stages][TILE_SIZE_M];
        float frag_b[num_ps2r_stages][TILE_SIZE_N];
        #pragma unroll
        for(int k = 0;k < BLOCK_SIZE_K;k ++){
            int ps2r_stage_id = k & 1;
            int next_ps2r_stage_id = (k + 1) & 1;

            if(k == 0){
                // load frag a
                #pragma unroll
                for(int frag_a_offset_m = 0; frag_a_offset_m < TILE_SIZE_M; frag_a_offset_m += num_float_per_fetch){
                    FLOAT4(frag_a[ps2r_stage_id][frag_a_offset_m]) = FLOAT4(As[pg2s_stage_id][k][tx * TILE_SIZE_M + frag_a_offset_m]);
                }
                // load frag b
                #pragma unroll
                for(int frag_b_offset_n = 0; frag_b_offset_n < TILE_SIZE_N; frag_b_offset_n += num_float_per_fetch){
                    FLOAT4(frag_b[ps2r_stage_id][frag_b_offset_n]) = FLOAT4(Bs[pg2s_stage_id][k][ty * TILE_SIZE_N + frag_b_offset_n]);
                }
            }
            if(k + 1 < BLOCK_SIZE_K){
                // preload frag a
                #pragma unroll
                for(int frag_a_offset_m = 0; frag_a_offset_m < TILE_SIZE_M; frag_a_offset_m += num_float_per_fetch){
                    FLOAT4(frag_a[next_ps2r_stage_id][frag_a_offset_m]) = FLOAT4(As[pg2s_stage_id][k + 1][tx * TILE_SIZE_M + frag_a_offset_m]);
                }
                // preload frag b
                #pragma unroll
                for(int frag_b_offset_n = 0; frag_b_offset_n < TILE_SIZE_N; frag_b_offset_n += num_float_per_fetch){
                    FLOAT4(frag_b[next_ps2r_stage_id][frag_b_offset_n]) = FLOAT4(Bs[pg2s_stage_id][k + 1][ty * TILE_SIZE_N + frag_b_offset_n]);
                }
            }
            // compute
            #pragma unroll
            for(int i = 0;i < TILE_SIZE_M;i ++){
                #pragma unroll
                for(int j = 0;j < TILE_SIZE_N;j ++){
                    sum[i][j] += frag_a[ps2r_stage_id][i] * frag_b[ps2r_stage_id][j];
                }
            }
        }
        if(bk + 1 < NUM_BLOCK_K){
            #pragma unroll
            for(int ldg_a_offset_m = 0;ldg_a_offset_m < BLOCK_SIZE_M;ldg_a_offset_m += ldg_block_a_stride){
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 0][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[0];
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 1][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[1];
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 2][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[2];
                As[next_pg2s_stage_id][tid % ldg_block_a_threads_per_row * num_float_per_fetch + 3][ldg_a_offset_m + tid / ldg_block_a_threads_per_row] = ldg_a_buffer[3];
            }
            __syncthreads();
        }
    }
    // store tile c
    #pragma unroll
    for(int i = 0;i < TILE_SIZE_M;i ++){
        #pragma unroll
        for(int j = 0;j < TILE_SIZE_N;j += num_float_per_fetch){
            FLOAT4(C[OFFSET(bm * BLOCK_SIZE_M + tx * TILE_SIZE_M + i, bn * BLOCK_SIZE_N + ty * TILE_SIZE_N + j, N)]) = FLOAT4(sum[i][j]);
        }
    }
}

torch::Tensor sgemm_smem_reg_coalesce_pg2s_ps2r(torch::Tensor A, torch::Tensor B){
    CHECK(A.dim() == 2, "A matrix dim should be 2");
    CHECK(B.dim() == 2, "B matrix dim should be 2");
    CHECK(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK(A.size(0) % 128 == 0, "gemm smem only support M mod 128 == 0")
    CHECK(A.size(1) % 8 == 0, "gemm smem only support K mod 8 == 0")
    CHECK(B.size(1) % 128 == 0, "gemm smem only support N mod 128 == 0")
    CHECK(A.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(B.scalar_type() == at::ScalarType::Float, "gemm smem only support float");
    CHECK(A.scalar_type() == B.scalar_type(), "gemm smem matrix A and B should have same type");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(A.scalar_type(), c_type, [&]{
        const int BLOCK_SIZE_M = 128;
        const int BLOCK_SIZE_K = 8;
        const int BLOCK_SIZE_N = 128;
        const int NUM_BLOCK_M = M / BLOCK_SIZE_M;
        const int NUM_BLOCK_N = N / BLOCK_SIZE_N;

        const int TILE_SIZE_M = 8;
        const int TILE_SIZE_N = 8;
        const int NUM_TILE_M = BLOCK_SIZE_M / TILE_SIZE_M;
        const int NUM_SIZE_N = BLOCK_SIZE_N / TILE_SIZE_N;
        dim3 gridDim(NUM_BLOCK_M, NUM_BLOCK_N);
        dim3 blockDim(NUM_TILE_M, NUM_SIZE_N);
        sgemm_smem_reg_coalesce_pg2s_ps2r_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, TILE_SIZE_M, TILE_SIZE_N><<<gridDim, blockDim>>>(
            static_cast<float*>(A.data_ptr()), 
            static_cast<float*>(B.data_ptr()), 
            static_cast<float*>(C.data_ptr()),
            M, K, N);
        return true;
    });
    return C;
}