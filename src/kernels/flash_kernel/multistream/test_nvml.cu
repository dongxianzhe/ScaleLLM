#include<nvml.h>
#include<iostream>

int main(){
    nvmlReturn_t result;
    result = nvmlInit();

    unsigned int deviceCount;
    result = nvmlDeviceGetCount(&deviceCount);
    if(NVML_SUCCESS != result){
        printf("failed to get device count\n");
        nvmlShutdown();
        return 1;
    }

    printf("deviceCount = %d\n", deviceCount);
    for(int i = 0;i < deviceCount;i ++){
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if(NVML_SUCCESS != result){
            printf("failed to get handle for device %d\n", i);
            continue;
        }
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if(NVML_SUCCESS != result){
            printf("failed to get utilization for device %d\n", i);
            continue;
        }
        std::cout << "GPU " << i << "compute utilization: " << utilization.gpu << "%" << std::endl;
        std::cout << "GPU " << i << "memory  utilization: " << utilization.memory << "%" << std::endl;
    }
}