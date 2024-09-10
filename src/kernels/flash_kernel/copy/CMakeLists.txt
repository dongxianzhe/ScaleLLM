include(cc_binary)

cc_binary(
  NAME
    test_copy
  SRCS
    test_copy.cu
  COPTS
    -lineinfo
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_tensor_copy
  SRCS
    test_tensor_copy.cu
  DEPS
    torch
    cutlass
)
