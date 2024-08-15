#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>
class Scheduler {
public:
    Scheduler(int streamsPerGpu = 2);
    ~Scheduler();
    void scheduleKernel(void (*kernel)(), bool isTensorCore);

    friend std::ostream& operator<<(std::ostream& os, const Scheduler& scheduler);
private:
    int numGpus;
    int streamsPerGpu;
    int highPriority;
    int lowPriority;
    std::vector<std::vector<cudaStream_t>> streams;
    void initStreams();
    int selectBestGPU();
    void cleanup();
};

Scheduler::Scheduler(int streamsPerGpu) : streamsPerGpu(streamsPerGpu) {
    cudaError_t cudaStatus = cudaGetDeviceCount(&numGpus);
    if (cudaStatus != cudaSuccess || numGpus <= 0) {
        throw std::runtime_error("No CUDA-enabled devices detected or failed to get device count.");
    }
    cudaStatus = cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to get stream priority range.");
    }
    streams.resize(numGpus, std::vector<cudaStream_t>(streamsPerGpu));
    initStreams();
}
Scheduler::~Scheduler() {
    cleanup();
}
void Scheduler::initStreams() {
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        // Create high priority stream for Tensor Core kernels
        cudaStreamCreateWithPriority(&streams[i][0], cudaStreamDefault, highPriority);
        // Create low priority stream for CUDA Core kernels
        cudaStreamCreateWithPriority(&streams[i][1], cudaStreamDefault, lowPriority);
    }
}
// Select the best GPU based on current load
int Scheduler::selectBestGPU() {
    int bestGPU = 0;
    float minUsage = std::numeric_limits<float>::max();
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        // Calculate memory usage percentage
        float usage = (totalMem - freeMem) / (float)totalMem * 100.0f;
        if (usage < minUsage) {
            minUsage = usage;
            bestGPU = i;
        }
    }
    return bestGPU;
}
// Schedule a kernel on the best GPU and stream
void Scheduler::scheduleKernel(void (*kernel)(), bool isTensorCore) {
    int gpu = selectBestGPU();
    cudaSetDevice(gpu);
    int streamIndex = isTensorCore ? 0 : 1;
    cudaStream_t stream = streams[gpu][streamIndex];

    // Launch kernel
    kernel<<<1, 1, 0, stream>>>();
    cudaStreamSynchronize(stream);
}

void Scheduler::cleanup() {
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < streamsPerGpu; j++) {
            cudaStreamDestroy(streams[i][j]);
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Scheduler& scheduler) {
    os << "Scheduler Info:\n";
    os << "Number of GPUs: " << scheduler.numGpus << "\n";
    os << "Streams per GPU: " << scheduler.streamsPerGpu << "\n";
    for (int i = 0; i < scheduler.numGpus; ++i) {
        os << "GPU " << i << ":\n";
        for (int j = 0; j < scheduler.streamsPerGpu; ++j) {
            os << " Stream " << j << ": " << scheduler.streams[i][j] << "\n"; 
        } 
    }
    return os; 
}

int main(){
    Scheduler scheduler;
    std::cout << scheduler << std::endl;
}