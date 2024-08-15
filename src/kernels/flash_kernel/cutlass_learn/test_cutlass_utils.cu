#include <cutlass/cutlass.h>
#include <cutlass/core_io.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm.h>
// Defines operator<<() to write TensorView objects to std::ostream
#include <cutlass/util/tensor_view_io.h>
// Defines cutlass::HostTensor<>
#include <cutlass/util/host_tensor.h>
// Defines cutlass::half_t
#include <cutlass/numeric_types.h>
// Defines device_memory::copy_device_to_device()
#include <cutlass/util/device_memory.h>
// Defines cutlass::reference::device::TensorFillRandomGaussian()
#include <cutlass/util/reference/device/tensor_fill.h>
// Defines cutlass::reference::host::TensorEquals()
#include <cutlass/util/reference/host/tensor_compare.h>

// Defines cutlass::reference::host::Gemm()
#include "cutlass/util/reference/host/gemm.h"


int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 256;
    // M-by-K matrix of cutlass::half_t
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));

    // K-by-N matrix of cutlass::half_t
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));

    // M-by-N matrix of cutlass::half_t
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M, N));

    // M-by-N matrix of cutlass::half_t
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M, N));


    // Arbitrary RNG seed value. Hard-coded for deterministic results.
    uint64_t seed = 2080;
    // Gaussian random distribution
    cutlass::half_t mean = 0.0_hf;
    cutlass::half_t stddev = 5.0_hf;
    // Specify the number of bits right of the binary decimal that are permitted
    // to be non-zero. A value of "0" here truncates random values to integers
    int bits_less_than_one = 0;
    cutlass::reference::device::TensorFillRandomGaussian(
        A.device_view(),
        seed,
        mean,
        stddev,
        bits_less_than_one
    );

    cutlass::reference::device::TensorFillRandomGaussian(
        B.device_view(),
        seed * 2019,
        mean,
        stddev,
        bits_less_than_one
    );
    
    cutlass::reference::device::TensorFillRandomGaussian(
        C_cutlass.device_view(),
        seed * 1993,
        mean,
        stddev,
        bits_less_than_one
    );

}