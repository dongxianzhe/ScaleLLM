include(cc_binary)

cc_binary(
  NAME
    test_inc
  SRCS
    test_inc.cu
  COPTS
    -lineinfo
  DEPS
    torch
)
