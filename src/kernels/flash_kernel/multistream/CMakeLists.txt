include(cc_binary)

cc_binary(
  NAME
    cuengine
  SRCS
    cuengine.cu
)

cc_binary(
  NAME
    cudagraphevent
  SRCS
    cudagraphevent.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    device_count
  SRCS
    device_count.cu
)

cc_binary(
  NAME
    stream_priority
  SRCS
    stream_priority.cu
)

cc_binary(
  NAME
    runtimeinfo
  SRCS
    runtimeinfo.cu
)

cc_binary(
  NAME
    multistream_scheduler1
  SRCS
    multistream_scheduler1.cu
  DEPS
    torch
    cutlass
  COPTS
    -I/usr/local/cuda/extras/CUPTI/include
    -I/usr/local/cuda/include
  LINKOPTS
    -L/usr/local/cuda/extras/CUPTI/lib64
    -lcupti
    -L/usr/local/cuda/lib64
    -lnvidia-ml
)

cc_binary(
  NAME
    test_cupti
  SRCS
    test_cupti.cu
  COPTS
    -I/usr/local/cuda/extras/CUPTI/include
  LINKOPTS
    -L/usr/local/cuda/extras/CUPTI/lib64
    -lcupti
)

cc_binary(
  NAME
    test_nvml
  SRCS
    test_nvml.cu
  COPTS
    -I/usr/local/cuda/extras/CUPTI/include
    -I/usr/local/cuda/include
  LINKOPTS
    -L/usr/local/cuda/extras/CUPTI/lib64
    -lcupti
    -L/usr/local/cuda/lib64
    -lnvidia-ml
)

cc_binary(
  NAME
    test_utilization
  SRCS
    test_utilization.cu
  COPTS
    -I/usr/local/cuda/extras/CUPTI/include
    -I/usr/local/cuda/include
  LINKOPTS
    -L/usr/local/cuda/extras/CUPTI/lib64
    -lcupti
    -L/usr/local/cuda/lib64
    -lnvidia-ml
)

# cc_binary(
#   NAME
#     test_flashinfer_kernels
#   SRCS
#     test_flashinfer_kernels.cu
#   COPTS
#     -I/usr/local/cuda/extras/CUPTI/include
#     -I/usr/local/cuda/include
#   LINKOPTS
#     -L/usr/local/cuda/extras/CUPTI/lib64
#     -lcupti
#     -L/usr/local/cuda/lib64
#     -lnvidia-ml
#   DEPS
#     torch
#     cutlass
#     flash_infer.kernels
#     # flashinfer
# )