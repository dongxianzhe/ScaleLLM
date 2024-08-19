#include<iostream>
#include"simulate_kernels.cuh"

void test_busy_wait_kernel(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    auto time_ms = ms2numclock(1.0);
    busy_wait_kernel<<<1, 128, 0>>>(time_ms);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("kernel execution %f ms\n", milliseconds);
}

int main(){
    test_busy_wait_kernel();
}