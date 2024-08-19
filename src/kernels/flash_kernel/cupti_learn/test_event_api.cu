// #include <cuda_runtime.h>
// #include <cupti.h>
// #include <stdio.h>
// void check(CUptiResult result, const char *func) {
//     if (result != CUPTI_SUCCESS) {
//         const char *errstr;
//         cuptiGetResultString(result, &errstr);
//         fprintf(stderr, "%s failed with error: %s\n", func, errstr);
//         exit(-1);
//     }
// }

// int main() {
//     CUdevice device;
//     CUpti_EventGroup eventGroup;
//     CUpti_EventID eventId;
//     uint64_t eventValue;
//     size_t bytesRead;
//     CUptiResult cuptiResult;
//     // 初始化 CUDA 设备
//     cuInit(0);
//     cuDeviceGet(&device, 0); // 获取设备 0
//     // 创建事件组
//     cuptiResult = cuptiEventGroupCreate(device, &eventGroup, 0);
//     check(cuptiResult, "cuptiEventGroupCreate");
//     // 获取事件 ID
//     cuptiResult = cuptiEventGetIdFromName(device, "active_warps", &eventId);
//     check(cuptiResult, "cuptiEventGetIdFromName");
//     // 将事件添加到事件组
//     cuptiResult = cuptiEventGroupAddEvent(eventGroup, eventId);
//     check(cuptiResult, "cuptiEventGroupAddEvent");
//     // 启用事件组
//     cuptiResult = cuptiEventGroupEnable(eventGroup);
//     check(cuptiResult, "cuptiEventGroupEnable");
//     // 执行 CUDA 代码
//     // 在这里调用 CUDA 内核或执行感兴趣的 CUDA 代码
//     // 读取事件计数
//     cuptiResult = cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE, eventId, &bytesRead, &eventValue);
//     check(cuptiResult, "cuptiEventGroupReadEvent");
//     printf("Active Warps: %llu\n", (unsigned long long)eventValue);
//     // 禁用事件组
//     cuptiResult = cuptiEventGroupDisable(eventGroup);
//     check(cuptiResult, "cuptiEventGroupDisable");
//     // 销毁事件组
//     cuptiResult = cuptiEventGroupDestroy(eventGroup);
//     check(cuptiResult, "cuptiEventGroupDestroy");
//     return 0;
// }

int main(){

}