#include<iostream>
#include<vector>

int main(){
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    printf("num_devices = %d \n", num_devices);
    std::vector<cudaStream_t> streams(num_devices);
    for(int i = 0;i < num_devices;i ++){
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    return 0;
}