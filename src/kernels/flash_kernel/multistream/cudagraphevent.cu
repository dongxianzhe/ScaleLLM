#include<iostream>
#include<torch/torch.h>

template<int block_size>
__global__ void inc_kenrel(float* aptr, float* bptr, int n){
    int i =blockIdx.x * block_size + threadIdx.x;
    if(i < n)bptr[i] = aptr[i] + 1;
}

void inc(torch::Tensor& a, torch::Tensor& b){
    const int n = a.size(0);
    const int block_size = 1024;
    dim3 gridDim((n + block_size - 1) / block_size);
    dim3 blockDim(block_size);
    inc_kenrel<block_size><<<gridDim, block_size, 0, 0>>>(
        static_cast<float*>(a.data_ptr()), 
        static_cast<float*>(b.data_ptr()), 
        n
    );
}

int main(){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int n = 2048;
    auto a = torch::zeros({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto b = torch::zeros({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // performance test
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    for(int i = 0;i < 1000;i ++){
        if(!graphCreated){
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(int j = 0;j < 20;j ++){
                const int block_size = 1024;
                dim3 gridDim((n + block_size - 1) / block_size);
                dim3 blockDim(block_size);
                inc_kenrel<block_size><<<gridDim, block_size, 0, stream>>>(
                    static_cast<float*>(a.data_ptr()), 
                    static_cast<float*>(b.data_ptr()), 
                    n
                );
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("kernel execution %f\n", milliseconds);


    // correctness test
    auto b_ref = a + 1;
    const float eps = 1e-3;
    if(torch::allclose(b_ref, b, eps, eps))puts("passed");
    else puts("failed");

    // puts("---------- b_ref --------------------");
    // std::cout << b_ref.slice(0, 0, 16) << std::endl;
    // puts("---------- b --------------------");
    // std::cout << b.slice(0, 0, 16) << std::endl;

    return 0;
}