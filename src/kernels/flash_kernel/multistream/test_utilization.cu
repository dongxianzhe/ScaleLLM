#include<iostream>
#include"simulate_kernels.cuh"
#include<nvml.h>
#include<unistd.h>
#include<cuda_runtime.h>

__global__ void cuda_core_kernel(long long num_clocks){
    long long start = clock64();

    volatile int x = 0;
    while((clock64() - start) < num_clocks){
        for(int i = 0;i < 1000000000;i ++){
            x += 1;
        }
    };
}


int main(){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "multiProcessorCount        : " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "concurrentKernels          : " << deviceProp.concurrentKernels << std::endl;
    std::cout << "clockRate                  : " << deviceProp.clockRate << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "sharedMemPerMultiprocessor : " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    std::cout << "regsPerMultiprocessor      : " << deviceProp.regsPerMultiprocessor << std::endl;


    int numBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, cuda_core_kernel, 769, 0);
    std::cout << "cudaOccupancyMaxActiveBlocksPerMultiprocessor: " << numBlocks << std::endl;

    for(int i = 0;i < 1;i ++){
        printf("------------- %d -----------------\n", i);
        cudaStream_t stream[3];
        for(int i = 0;i < 3;i ++)cudaStreamCreate(&stream[i]);

        cudaEvent_t start[3], stop[3];
        for(int i = 0;i < 3;i ++){
            cudaEventCreate(&start[i]);
            cudaEventCreate(&stop[i]);
        }

        for(int i = 0;i < 3;i ++){
            cudaEventRecord(start[i], stream[i]);
        }

        // cuda_core_kernel<<<28, 768, 0, stream[1]>>>(ms2numclock(1000));
        // cuda_core_kernel<<<28, 768, 0, stream[1]>>>(ms2numclock(1000));
        cuda_core_kernel<<<57, 769, 0, stream[2]>>>(ms2numclock(1000));

        cudaEventRecord(stop[1], stream[1]);
        cudaEventRecord(stop[2], stream[2]);

        cudaStreamSynchronize(stream[1]);
        cudaStreamSynchronize(stream[2]);
        cudaEventRecord(stop[0], stream[0]);

        for(int i = 0;i < 3;i ++){
            cudaEventSynchronize(stop[i]);
        }
        // nvmlInit();
        // nvmlDevice_t device;
        // nvmlDeviceGetHandleByIndex(0, &device);
        // nvmlUtilization_t utilization;
        // nvmlDeviceGetUtilizationRates(device, &utilization);
        // nvmlShutdown();
        // printf("utilization %d\n", utilization.gpu);

        for(int i = 0;i < 3;i ++){
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start[i], stop[i]);
            printf("meansure %d %f ms\n", i, milliseconds);
        }

        for(int i = 0;i < 3;i ++){
            cudaStreamDestroy(stream[i]);
        }
        sleep(3);
    }
}