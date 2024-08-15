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
    multistream_scheduler
  SRCS
    multistream_scheduler.cu
)