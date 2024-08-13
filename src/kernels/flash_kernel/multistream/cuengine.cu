#include<iostream>
#include<cuda_runtime.h>

int main(){
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("prop->asyncEngineCount = %d\n", prop.asyncEngineCount);

    // printf("", cudaGetDevice(0)):
}