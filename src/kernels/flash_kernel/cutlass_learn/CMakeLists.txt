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

cc_binary(
  NAME
    test_cutlass_utils
  SRCS
    test_cutlass_utils.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_dump_reg_shmem_
  SRCS
    test_dump_reg_shmem_.cu
  DEPS
    cutlass
)

cc_binary(
  NAME
    test_batch_gemm
  SRCS
    test_batch_gemm.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_splitk_gemm
  SRCS
    test_splitk_gemm.cu
  DEPS
    torch
    cutlass
)