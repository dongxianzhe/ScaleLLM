#pragma once

#define CHECK_FATAL(predicate, message) { \
    if(!(predicate)){                     \
        fprintf(stderr, "%s", (message)); \
        exit(1);                          \
    }                                     \
}

#define CHECK(predicate, message) {       \
    if(!(predicate)){                     \
        fprintf(stderr, "%s", (message)); \
        exit(1);                          \
    }                                     \
}

#define checkCudaErrors(func)                                                                  \
{                                                                                              \
    cudaError_t e = (func);                                                                    \
    if(e != cudaSuccess)printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
}

#define div_ceil(x, mod) (((x) + (mod) - 1) / (mod))
#define OFFSET(row, col, stride) (((row) * (stride)) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])