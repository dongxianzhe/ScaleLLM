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
