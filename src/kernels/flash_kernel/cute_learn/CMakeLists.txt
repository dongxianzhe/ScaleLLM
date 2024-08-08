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

cc_binary(
  NAME
    test_swizzle
  SRCS
    test_swizzle.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_cute_copy
  SRCS
    test_cute_copy.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_cp_async
  SRCS
    test_cp_async.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_zip_divide
  SRCS
    test_zip_divide.cu
  DEPS
    cutlass
)

cc_binary(
  NAME
    test_cute_gemm_algorithm
  SRCS
    test_cute_gemm_algorithm.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_mma
  SRCS
    test_mma.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_tile_to_shape
  SRCS
    test_tile_to_shape.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_mma_inc
  SRCS
    test_mma_inc.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_print_latex
  SRCS
    test_print_latex.cu
  DEPS
    cutlass
)

cc_binary(
  NAME
    test_ldmatrix
  SRCS
    test_ldmatrix.cu
  DEPS
    torch
    cutlass
)

cc_binary(
  NAME
    test_2gemm
  SRCS
    test_2gemm.cu
  DEPS
    torch
    cutlass
)