include(cc_binary)

cc_binary(
  NAME
    test_add
  SRCS
    add.cuh
    test_add.cu
  DEPS
    torch
)

cc_binary(
  NAME
    test_sum
  SRCS
    sum.cuh
    test_sum.cu
  DEPS
    torch
)

cc_binary(
  NAME
    test_gemm
  SRCS
    sgemm.cuh
    test_gemm.cu
  DEPS
    torch
)

cc_binary(
  NAME
    test_single_decode
  SRCS
    single_decode.cuh
    test_single_decode.cu
  DEPS
    torch
)

cc_binary(
  NAME
    test_cute_layout
  SRCS
    test_cute_layout.cu
  DEPS
    torch
    
)