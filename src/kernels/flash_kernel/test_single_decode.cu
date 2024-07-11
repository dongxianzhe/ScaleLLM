#include <torch/torch.h>
#include <iostream>
#include"single_decode.cuh"

// Apply rotary position embedding function
torch::Tensor apply_rotary_position_embedding(torch::Tensor x, torch::Tensor offset) {
    // x: (seq_len, ..., head_dim)
    // offset: (seq_len, ...)
    int64_t head_dim = x.size(-1);
    auto device = x.device();
    auto inv_freq = 1.0 / torch::pow(10000.0, torch::arange(0, head_dim, 2, torch::TensorOptions().dtype(torch::kFloat).device(device)) / head_dim); // (head_dim / 2, )
    auto w = inv_freq * offset.unsqueeze(-1); // (seq_len, ..., head_dim / 2)
    auto cosw = torch::cos(w);
    auto sinw = torch::sin(w);
    auto x1 = cosw * x.slice(-1, 0, head_dim / 2) - sinw * x.slice(-1, head_dim / 2, head_dim);
    auto x2 = sinw * x.slice(-1, 0, head_dim / 2) + cosw * x.slice(-1, head_dim / 2, head_dim);
    return torch::cat({x1, x2}, -1);
}

// Single decode function
torch::Tensor single_decode_with_kv_cache_ref(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    /**
    * :param q: (num_qo_head, head_dim)
    * :param k: (seq_len, num_kv_head, head_dim)
    * :param v: (seq_len, num_kv_head, head_dim)
    * :return: o (num_q_head * head_dim)
    */
    auto device = q.device();
    int64_t num_qo_head = q.size(0);
    int64_t head_dim = q.size(1);
    int64_t seq_len = k.size(0);
    int64_t num_kv_head = k.size(1);
    int64_t group_size = num_qo_head / num_kv_head;

    auto q_offset = torch::tensor({static_cast<float>(seq_len - 1)}, torch::TensorOptions().dtype(torch::kFloat).device(device)).repeat(num_qo_head);
    auto k_offset = torch::arange(seq_len, torch::TensorOptions().dtype(torch::kFloat).device(device)).unsqueeze(1).repeat({1, num_kv_head});

    q = apply_rotary_position_embedding(q, q_offset);
    k = apply_rotary_position_embedding(k, k_offset);

    float sm_scale = std::sqrt(head_dim);
    q = q / sm_scale; // (num_qo_head, head_dim)
    k = k.permute({1, 0, 2}); // (num_kv_head, seq_len, head_dim)
    k = k.repeat_interleave(group_size, 0); // (num_qo_head, seq_len, head_dim)

    auto s = torch::matmul(q.unsqueeze(1), k.transpose(-1, -2)); // (num_qo_head, 1, seq_len)
    s = torch::softmax(s, -1); // (num_qo_head, 1, seq_len)

    v = v.permute({1, 0, 2}); // (num_kv_head, seq_len, head_dim)
    v = v.repeat_interleave(group_size, 0); // (num_qo_head, seq_len, head_dim)

    auto o = torch::matmul(s, v); // (num_qo_head, 1, head_dim)
    return o.reshape({num_qo_head, head_dim});
}
void test(torch::Tensor a, torch::Tensor b, std::string name){
    if (a.is_cuda())a = a.to(torch::kCPU);
    if (b.is_cuda())b = b.to(torch::kCPU);
    float eps = 1e-3;
    if (a.allclose(b, eps, eps)) {
        std::cout << name << ": pass" << std::endl;
    } else {
        std::cout << name << ": fail" << std::endl;
    }
}

int main() {
    int num_qo_head = 32;
    int head_dim = 256;
    int seq_len = 512;
    int num_kv_head = 4;

    torch::Device device(torch::kCUDA);
    auto q = torch::randn({num_qo_head, head_dim}, torch::dtype(torch::kFloat).device(device));
    auto k = torch::randn({seq_len, num_kv_head, head_dim}, torch::dtype(torch::kFloat).device(device));
    auto v = torch::randn({seq_len, num_kv_head, head_dim}, torch::dtype(torch::kFloat).device(device));

    auto o = single_decode_with_kv_cache(q, k, v);

    auto o_ref = single_decode_with_kv_cache_ref(q, k, v);
    std::cout << o_ref.sizes() << std::endl;

    test(o, o_ref, "single_decode_with_kv_cache");

    torch::Tensor sliced_tensor;
    sliced_tensor = o.index({torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    std::cout << sliced_tensor << std::endl;
    sliced_tensor = o_ref.index({torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    std::cout << sliced_tensor << std::endl;


    return 0;
}
