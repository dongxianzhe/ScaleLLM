#include<cuda_runtime.h>
__global__ void busy_wait_kernel(long long num_clocks){
    long long start = clock64();

    volatile int x = 0;
    while((clock64() - start) < num_clocks){
        for(int i = 0;i < 100000000;i ++){
            x += 1;
        }
    };
}


long long ms2numclock(float time_ms){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    auto clockRatekHz = deviceProp.clockRate;
    return time_ms * clockRatekHz;
}
