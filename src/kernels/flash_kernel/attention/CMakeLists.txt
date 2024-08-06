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

cc_binary(
  NAME
    prefill
  SRCS
    prefill.cu
  DEPS
    torch
    cutlass
)