include(cc_binary)

cc_binary(
  NAME
    hello
  SRCS
    hello.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    gemm
  SRCS
    gemm.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    basic_gemm
  SRCS
    basic_gemm.cu
  DEPS
    torch
    cutlass
)