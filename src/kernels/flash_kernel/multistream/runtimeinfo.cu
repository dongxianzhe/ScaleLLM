#include<cuda_runtime.h>
#include<iostream>
#include<vector>

int main(){
    // nvmlInit();
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);

    // std::vector<nvmlDevice_t> devices(deviceCount);
    // std::vector<nvmlMemory_t> memInfos(deviceCount);

    // for(int i = 0;i < deviceCount;i ++){
    //     nvmlDevice_t device;
    //     nvmlDeviceGetHandleByIndex(i, &device);
    //     devices[i] = device;

    //     nvmlMemory_t memInfo;
    //     nvmlDeviceGetMemoryInfo(device, &memInfo);
    //     memInfos[i] = memInfo;
    // }

    // for(int i = 0;i < deviceCount;i ++){
    //     nvmlMemory_t& memInfo = memInfos[i];
    //     std::cout << "device " << i << "memory info: " << std::endl;
    //     std::cout << "total memory: " << memInfo.total / 1024 / 1024 << " MB" << std::endl;
    //     std::cout << "free  memory: " << memInfo.free  / 1024 / 1024 << " MB" << std::endl;
    //     std::cout << "used  memory: " << memInfo.used  / 1024 / 1024 << " MB" << std::endl;
    // }

    // nvmlShutdown();
    // return 0;

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
}