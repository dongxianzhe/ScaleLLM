include(cc_library)

cc_library(
  NAME 
    cutlass
  INCLUDES
    cutlass/include
  DEPS 
    torch # TODO: depends on CUDA instead of torch
)

cc_library(
  NAME 
    flashinfer
  INCLUDES
    flashinfer/include
)

add_subdirectory(sentencepiece)
add_subdirectory(nvbench)
# add_subdirectory(jinja2cpp)

