#include<cuda_runtime.h>
#include<iostream>
#include<vector>

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::vector<cudaDeviceProp> deviceProps(deviceCount);
    std::vector<size_t>freeMemory(deviceCount);
    std::vector<size_t>totalMemory(deviceCount);

    for(int i = 0;i < deviceCount;i ++){
        cudaSetDevice(i);

        cudaMemGetInfo(&freeMemory[i], &totalMemory[i]);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        deviceProps[i] = prop;
    }

    for(int i = 0;i < deviceCount;i ++){
        const cudaDeviceProp& prop = deviceProps[i];
        std::cout << "device " << i << "(" << prop.name << ")" << "memory info: " << std::endl;
        std::cout << "total memory: " << totalMemory[i] / 1024 / 1024 << " MB" << std::endl;
        std::cout << "free  memory: " << freeMemory[i]  / 1024 / 1024 << " MB" << std::endl;
        std::cout << "used  memory: " << (totalMemory[i] - freeMemory[i]) / 1024 / 1024 << " MB" << std::endl;
    }
    // prop
    std::cout << "---------- prop --------------------" << std::endl;
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "totalGlobalMem:            " << prop.totalGlobalMem / 1024 / 1024 << " GB" << std::endl;
        std::cout << "sharedMemPerBlock:         " << prop.sharedMemPerBlock << std::endl;
        std::cout << "regsPerBlock               " << prop.regsPerBlock << std::endl;
        std::cout << "warpSize                   " << prop.warpSize << std::endl;
        std::cout << "maxThreadsPerBlock         " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "multiProcessorCount        " << prop.multiProcessorCount << std::endl;
        std::cout << "kernelExecTimeoutEnabled   " << prop.kernelExecTimeoutEnabled << std::endl;
        std::cout << "concurrentKernels          " << prop.concurrentKernels << std::endl;
        std::cout << "asyncEngineCount           " << prop.asyncEngineCount << std::endl;
        std::cout << "l2CacheSize                " << prop.l2CacheSize << std::endl;
        std::cout << "maxThreadsPerMultiProcessor" << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "streamPrioritiesSupported  " << prop.streamPrioritiesSupported << std::endl;
        std::cout << "sharedMemPerMultiprocessor " << prop.sharedMemPerMultiprocessor << std::endl;
        std::cout << "regsPerMultiprocessor      " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "sharedMemPerBlockOptin     " << prop.sharedMemPerBlockOptin << std::endl;
        std::cout << "maxBlocksPerMultiProcessor " << prop.maxBlocksPerMultiProcessor << std::endl;




    }
}