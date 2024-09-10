#include<iostream>
#include"flash_api.h"

int main(){
    const int n_tokens = 64;
    const int n_heads = 32;
    const int head_dim = 128;
    const int n_blocks = 3172;
    const int block_size = 16;
    const int n_kv_heads = 2;
    const int batch = 64;
    const int max_blocks_per_seq = 129;
    // out [n_tokens, n_heads, head_dim]
    // q [n_tokens, n_heads, head_dim]
    // k [n_blocks, block_size, n_kv_heads, head_dim] 
    // v [n_blocks, block_size, n_kv_heads, head_dim] 
    // cu_seqlens_q [batch + 1]
    // cu_seqlens_k [batch + 1]
    // block_table_ [batch, max_blocks_per_seq]
    // cu_block_lens [batch + 1]
    at::Tensor out = torch::randn({n_tokens, n_heads, head_dim}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    const at::Tensor q = torch::randn({n_tokens, n_heads, head_dim}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    const at::Tensor k = torch::randn({n_blocks, block_size, n_kv_heads, head_dim}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    const at::Tensor v = torch::randn({n_blocks, block_size, n_kv_heads, head_dim}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    const at::Tensor cu_seqlens_q = torch::zeros({batch + 1}, torch::dtype(torch::kInt).device(torch::kCUDA));
    const at::Tensor cu_seqlens_k = torch::zeros({batch + 1}, torch::dtype(torch::kInt).device(torch::kCUDA));
    c10::optional<at::Tensor> block_table_ = torch::zeros({batch, max_blocks_per_seq}, torch::dtype(torch::kInt).device(torch::kCUDA));
    c10::optional<at::Tensor> cu_block_lens = torch::zeros({batch + 1}, torch::dtype(torch::kInt).device(torch::kCUDA));
    c10::optional<at::Tensor> alibi_slopes = c10::nullopt;

    int max_seqlen_q = 12;
    int max_seqlen_k = 1500;
    float softmax_scale = 0.088388;
    float softcap = 0.;
    int window_size_left = -1;
    int window_size_right = 0;
    int num_splits = 0;

    mha_varlen_fwd(
        out, 
        q, 
        k, 
        v, 
        cu_seqlens_q, 
        cu_seqlens_k, 
        block_table_, 
        cu_block_lens, 
        alibi_slopes, 
        max_seqlen_q, 
        max_seqlen_k, 
        softmax_scale, 
        softcap, 
        window_size_left, 
        window_size_right, 
        num_splits);

    std::cout << "test_flash_api finished" << std::endl;
    return 0;
}