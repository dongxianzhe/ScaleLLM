#include<iostream>
#include<cuda_fp16.h>

__global__ void kernel1(){
    volatile __shared__ half s[26624];
}

__global__ void kernel2(){
    volatile extern __shared__ half s[];
}

int main(){
    dim3 gridDim(8, 8);
    dim3 blockDim(128);
    kernel1<<<gridDim, blockDim>>>();
    kernel1<<<gridDim, blockDim>>>();
    return 0;
}