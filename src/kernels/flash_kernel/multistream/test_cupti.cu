#include <cupti.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unistd.h>

void CUPTIAPI getActivityCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {
    printf("callback domain = %d cbid = %d\n", domain, cbid);
    // if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        // if(cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020){
        // }
    // }
}


__global__ void kernel(float* a){
    int tid = threadIdx.x;
    if(tid < 100){
        a[tid] = a[tid] + 1;
    }
}

int main() {
    CUpti_SubscriberHandle subscriber;
    CUptiResult res = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getActivityCallback, nullptr);
    if (res != CUPTI_SUCCESS) {
        printf("Failed to initialize CUPTI.\n");
        return 0;
    }
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_INVALID           );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API        );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API       );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE          );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE       );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX              );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_STATE             );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SIZE              );
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_FORCE_INT         );
  
    // cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);


    // 在这里执行 CUDA 程序
    // 示例：启动一个简单的 kernel
    cudaFree(0);  // 初始化 CUDA 上下文
    float* d_a;
    cudaMalloc(&d_a, 100 * sizeof(float));
    // 执行其他 CUDA 操作或 kernel
    kernel<<<1, 64>>>(d_a);


    // 停止 CUPTI 并清理资源
    cuptiUnsubscribe(subscriber);
    return 0;
}