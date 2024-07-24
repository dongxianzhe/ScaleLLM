include(cc_binary)

cc_binary(
  NAME
    test_gemm1
  SRCS
    test_gemm.cu
  COPTS
    -lineinfo
  DEPS
    torch
    cutlass
)
