include(cc_library)
include(cc_binary)

# copy the file to binary dir in order to force re-configuration when the file changes
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py
  ${CMAKE_CURRENT_BINARY_DIR}/generate_instantiation_cu.py
)

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
)

# globbing all generated files in sub directory "generated"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/*.cu")

cc_library(
  NAME 
    flash_attn.kernels
  HDRS
    flash_api.h
  SRCS
    flash_api.cpp
    ${GENERATED_SRC_FILES}
  INCLUDES
    src
  DEPS
    cutlass
)

cc_binary(
  NAME 
    test_flash_api
  # HDRS
  #   flash_api.h
  SRCS
    test_flash_api.cpp
  DEPS
    torch
    flash_attn.kernels
)