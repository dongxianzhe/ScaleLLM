include(cc_binary)

cc_binary(
  NAME
    test_tensor
  SRCS
    test_tensor.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_tiledmma
  SRCS
    test_tiledmma.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_cute_gemm
  SRCS
    test_cute_gemm.cu
  DEPS
    torch
    cutlass
)