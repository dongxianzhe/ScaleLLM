#include<cuda_fp16.h>
#include<torch/torch.h>
#include"common.cuh"
#include"dispatch.cuh"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

__global__ void hgemm_naive_kernel(half *__restrict__ A, half *__restrict__ B, half *__restrict__ C, int M,
                               int K, int N) {
    const int K_tiles = div_ceil(K, MMA_K);

    const int warp_row = blockIdx.y * MMA_M;
    const int warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    const int lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (int i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);
        // A_smem一行是16个half是8个float，一次取四个float，需要两个线程取数。
        // lane_id / 2线程分配到的行数，lane_id % 2线程分配到的列数。 因为+之前已经转为int4类型了，所以+相当于+land_id % 2 * 4个字节
        // A[warp_row + lane_id / 2][i * MMA_K]

        if (lane_id < MMA_N * 2) {  // smem 一行需要2个，共需要MMA_N * 2个线程取数
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }
        // B_smem一行也是16个half，需要两个线程取数。
        // B 是按列优先存储的indices(i * MMA_K + lane_id % 2 * 4, warp_col + land_id / 2) stride (1, K)

        __syncthreads();

        uint32_t RA[4]; // 8个half
        uint32_t RB[2]; // 4个half

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}

torch::Tensor hgemm_naive(torch::Tensor A, torch::Tensor B) {
    CHECK(A.dim() == 2, "A matrix dim should be 2");
    CHECK(B.dim() == 2, "B matrix dim should be 2");
    CHECK(A.size(1) == B.size(0), "A.size(1) should be equal to B.size(0)");
    CHECK(A.scalar_type() == at::ScalarType::Half, "gemm smem only support float");
    CHECK(B.scalar_type() == at::ScalarType::Half, "gemm smem only support float");
    CHECK(A.scalar_type() == B.scalar_type(), "gemm smem matrix A and B should have same type");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty_like(A, A.options());

    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    hgemm_naive_kernel<<<grid, block>>>(
        static_cast<half*>(A.data_ptr()), 
        static_cast<half*>(B.data_ptr()), 
        static_cast<half*>(C.data_ptr()),
        M, K, N);
    return C;
}
