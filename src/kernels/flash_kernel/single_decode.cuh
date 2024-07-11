#include "common.cuh"
#include "dispatch.cuh"

#include <iostream>
#include <torch/torch.h>

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template<int group_size, int vec_size, int head_dim> 
__global__ void SingleDecodeWithKVCacheKernel(float* q, float* k, float* v, float* o, int num_qo_heads, int num_kv_heads, int seq_len){
    // // gridDim (num_kv_heads)
    // // blockDim (warp_size, group_size)
    // // q (num_qo_heads, head_dim)
    // // k,v (seq_len, num_kv_heads, head_dim)
    // // o (num_qo_heads, head_dim)
    int warp_size = 32;
    int warp_idx = threadIdx.y;
    int lane_idx = threadIdx.x % 32;
    int qo_head_idx_within_group = threadIdx.y;
    int kv_head_idx = blockIdx.x;

    int qo_head_idx = kv_head_idx * group_size + qo_head_idx_within_group;

    // 1. load q to register, q_vec[:] = q[qo_head_idx, lane_idx * vec_size: lane_idx * vec_size + vec_size]
    float q_vec[vec_size];
    float permute_vec[vec_size];
    float o_vec[vec_size] = {0};
    for(int i = 0;i < vec_size;i += 4){
        // q_vec[:] = q[qo_head_idx, lane_idx * vec_size + i: lane_idx * vec_size + i + 4]
        FLOAT4(q_vec[i]) = FLOAT4(q[qo_head_idx * head_dim + (lane_idx * vec_size + i)]);
    }
    // 2. rotate q
    if(lane_idx < warp_size / 2){
        for(int i = 0;i < vec_size;i += 4){
            FLOAT4(permute_vec[i]) = FLOAT4(q[qo_head_idx * head_dim + (lane_idx * vec_size + head_dim / 2 + i)]);
        }
    }else{
        for(int i = 0;i < vec_size;i += 4){
            FLOAT4(permute_vec[i]) = FLOAT4(q[qo_head_idx * head_dim + (lane_idx * vec_size - head_dim / 2 + i)]);
        }
    }
    for(int i = 0;i < vec_size;i ++){
        float freq = __powf(1. / 10000., float(2 * ((lane_idx * vec_size + i) % (head_dim / 2))) / float(head_dim));
        float w = (seq_len - 1) * freq;
        float cosw, sinw;
        __sincosf(w, &sinw, &cosw);
        if(lane_idx < warp_size / 2)q_vec[i] = cosw * q_vec[i] - sinw * permute_vec[i];
        else q_vec[i] = cosw * q_vec[i] + sinw * permute_vec[i];
    }
    // 3. scale q
    const float log2e = 1.44269504088896340736f;
    float sm_scale = 1. / sqrtf(float(head_dim)) * log2e;
    for(int i = 0;i < vec_size;i ++)q_vec[i] *= sm_scale;

    constexpr int KV_CHUNK_SIZE = group_size;
    float m = -1e5;
    float d = 0;
    for(int kv_token_idx = 0; kv_token_idx < seq_len; kv_token_idx += KV_CHUNK_SIZE){
        // 4. load k v block to shared memory
        __shared__ float ks[KV_CHUNK_SIZE][head_dim];// ks[:, :] = k[kv_token_idx: kv_token_idx + KV_CHUNK_SIZE, kv_head_idx, :];
        __shared__ float vs[KV_CHUNK_SIZE][head_dim];// ks[:, :] = k[kv_token_idx: kv_token_idx + KV_CHUNK_SIZE, kv_head_idx, :];
        // ks[warp_idx][lane_idx * vec_size : lane_idx * vec_size + vec_size] = k[kv_token_idx + warp_idx][kv_head_idx][lane_idx * vec_size : lane_idx * vec_size + vec_size]
        for(int i = 0;i < vec_size;i += 4){
            FLOAT4(ks[warp_idx][lane_idx * vec_size + i]) = FLOAT4(k[((kv_token_idx + warp_idx) * num_kv_heads + kv_head_idx) * head_dim + lane_idx * vec_size + i]);
        }
        // vs[warp_idx][lane_idx * vec_size : lane_idx * vec_size + vec_size] = v[kv_token_idx + warp_idx][kv_head_idx][lane_idx * vec_size : lane_idx * vec_size + vec_size]
        for(int i = 0;i < vec_size;i += 4){
            FLOAT4(vs[warp_idx][lane_idx * vec_size + i]) = FLOAT4(v[((kv_token_idx + warp_idx) * num_kv_heads + kv_head_idx) * head_dim + lane_idx * vec_size + i]);
        }
        __syncthreads();

        float scores[KV_CHUNK_SIZE];
        for(int token_idx_within_kv_chunk = 0; token_idx_within_kv_chunk < KV_CHUNK_SIZE; token_idx_within_kv_chunk ++){
            // 5. load k token to register, ks[token_idx_within_kv_chunk, lane_idx * vec_size: lane_idx * vec_size + vec_size]
            float k_vec[vec_size];
            for(int i = 0;i < vec_size;i += 4){
                FLOAT4(k_vec[i]) = FLOAT4(ks[token_idx_within_kv_chunk][lane_idx * vec_size + i]);
            }
            // 6. rotate k
            if(lane_idx < warp_size / 2){
                for(int i = 0;i < vec_size;i += 4){
                    FLOAT4(permute_vec[i]) = FLOAT4(ks[token_idx_within_kv_chunk][lane_idx * vec_size + head_dim / 2 + i]);
                }
            }else{
                for(int i = 0;i < vec_size;i += 4){
                    FLOAT4(permute_vec[i]) = FLOAT4(ks[token_idx_within_kv_chunk][lane_idx * vec_size - head_dim / 2 + i]);
                }
            }
            for(int i = 0;i < vec_size;i ++){
                float freq = __powf(1. / 10000., float(2 * ((lane_idx * vec_size + i) % (head_dim / 2))) / float(head_dim));
                float w = (kv_token_idx + token_idx_within_kv_chunk) * freq;
                float cosw, sinw;
                __sincosf(w, &sinw, &cosw);
                if(lane_idx < warp_size / 2)k_vec[i] = cosw * k_vec[i] - sinw * permute_vec[i];
                else k_vec[i] = cosw * k_vec[i] + sinw * permute_vec[i];
            }

            float s = 0;
            // 7. compute q k
            for(int i = 0;i < vec_size;i ++){
                s += q_vec[i] * k_vec[i];
            }
            // 8. warp reduce, caculate k token score
            for(int offset = warp_size / 2; offset > 0; offset /= 2){
                s += __shfl_xor_sync(0xffffffff, s, offset);
            }
            scores[token_idx_within_kv_chunk] = s;
        }
        // 9. update m
        float m_prev = m;
        for(int token_idx_within_kv_chunk = 0; token_idx_within_kv_chunk < KV_CHUNK_SIZE; token_idx_within_kv_chunk ++){
            m = max(m, scores[token_idx_within_kv_chunk]);
        }
        // 10. update d
        float d_prev = d;
        d = d_prev * ptx_exp2(m_prev - m);
        for(int token_idx_within_kv_chunk = 0; token_idx_within_kv_chunk < KV_CHUNK_SIZE; token_idx_within_kv_chunk ++){
            d += ptx_exp2(scores[token_idx_within_kv_chunk] - m);
        }
        // 11. power s
        for(int i = 0;i < KV_CHUNK_SIZE;i ++){
            scores[i] = ptx_exp2(scores[i] - m);
        }

        // 12. update o
        for(int i = 0;i < vec_size;i ++){
            o_vec[i] *= ptx_exp2(m_prev - m);
        }
        for(int token_idx_within_kv_chunk = 0; token_idx_within_kv_chunk < KV_CHUNK_SIZE; token_idx_within_kv_chunk ++){
            float v_vec[vec_size]; // 13. load v to register: v_vec[:] = vs[token_idx_within_kv_chunk][lane_idx * vec_size : lane_idx * vec_size + vec_size]
            for(int i = 0;i < vec_size;i += 4){
                FLOAT4(v_vec[i]) = FLOAT4(vs[token_idx_within_kv_chunk][lane_idx * vec_size + i]);
            }
            for(int i = 0;i < vec_size;i ++){
                o_vec[i] += scores[token_idx_within_kv_chunk] * v_vec[i];
            }
        }
        __syncthreads();
    }
    // 13. o divide by d
    for(int i = 0;i < vec_size;i ++){
        o_vec[i] /= d;
    }
    // 14. store o: o[qo_head_idx, lane_idx * vec_size: lane_idx * vec_size + vec_size] = o_vec[:];
    for(int i = 0;i < vec_size;i += 4){
        FLOAT4(o[qo_head_idx * head_dim + lane_idx * vec_size + i]) = FLOAT4(o_vec[i]);
    }
}

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v){
    // q (num_qo_heads, head_dim)
    // k (kv_len, num_kv_heads, head_dim)
    // v (kv_len, num_kv_heads, head_dim)
    CHECK(q.dim() == 2, "q shape should be (num_qo_heads, head_dim)");
    CHECK(k.dim() == 3, "k shape should be (kv_len, num_kv_heads, head_dim)");
    CHECK(v.dim() == 3, "v shape should be (kv_len, num_kv_heads, head_dim)");
    CHECK(k.size(0) == v.size(0), "k and v should have same kv_len");
    CHECK(k.size(1) == v.size(1), "k and v should have same num_kv_heads");
    CHECK(k.size(2) == v.size(2), "k and v should have same head_dim");
    CHECK(q.size(0) % k.size(1) == 0, "num_qo_heads % num_kv_heads should be zero");
    CHECK(q.scalar_type() == at::ScalarType::Float, "temporary only support float");
    CHECK(k.scalar_type() == at::ScalarType::Float, "temporary only support float");
    CHECK(v.scalar_type() == at::ScalarType::Float, "temporary only support float");

    int num_qo_heads = q.size(0);
    int seq_len = k.size(0);
    int num_kv_heads = k.size(1);

    auto o = torch::empty_like(q, q.options());

    using c_type = float;
    DISPATCH_group_size(num_qo_heads / num_kv_heads, group_size, [&] {
        return DISPATCH_head_dim(q.size(1), head_dim, [&] {
            constexpr int warp_size = 32;
            dim3 gridDim(num_kv_heads);
            dim3 blockDim(warp_size, group_size);
            SingleDecodeWithKVCacheKernel<group_size, head_dim / warp_size, head_dim><<<gridDim, blockDim>>>(
                static_cast<c_type*>(q.data_ptr()),
                static_cast<c_type*>(k.data_ptr()),
                static_cast<c_type*>(v.data_ptr()),
                static_cast<c_type*>(o.data_ptr()),
                num_qo_heads,
                num_kv_heads,
                seq_len
            );
            return true;
        });
    });

    return o;
}