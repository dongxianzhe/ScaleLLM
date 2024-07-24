#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <numeric>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
using namespace cute;

__global__ void
test(double const* g_in, double* g_out){
  extern __shared__ double smem[];
  smem[threadIdx.x] = g_in[threadIdx.x];
  __syncthreads();
  g_out[threadIdx.x] = 2 * smem[threadIdx.x];
}

__global__ void
test2(double const* g_in, double* g_out){
  using namespace cute;
  extern __shared__ double smem[];

  auto s_tensor = make_tensor(make_smem_ptr(smem + threadIdx.x), Int<1>{});
  auto g_tensor = make_tensor(make_gmem_ptr(g_in + threadIdx.x), Int<1>{});

  copy(g_tensor, s_tensor);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  g_out[threadIdx.x] = 2 * smem[threadIdx.x];
}

int main(){
    constexpr int count = 32;
    thrust::host_vector<double> h_in(count);
    for (int i = 0; i < count; ++i) {
    h_in[i] = double(i);
    }

    thrust::device_vector<double> d_in(h_in);

    thrust::device_vector<double> d_out(count, -1);
    test<<<1, count, sizeof(double) * count>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
    thrust::host_vector<double> h_result = d_out;

    thrust::device_vector<double> d_out_cp_async(count, -2);
    test2<<<1, count, sizeof(double) * count>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out_cp_async.data()));
    thrust::host_vector<double> h_result_cp_async = d_out_cp_async;


    const double eps = 1e-3;
    for(int i = 0;i < count;i ++){
        if(abs(h_result[i] - h_result_cp_async[i]) > eps){
            puts("fail");
            return 0;
        }
    }
    puts("pass");
    return 0;
}