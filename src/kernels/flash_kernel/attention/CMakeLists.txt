include(cc_binary)

cc_binary(
  NAME
    test_attention
  SRCS
    test_attention.cu
  DEPS
    torch
    cutlass
)