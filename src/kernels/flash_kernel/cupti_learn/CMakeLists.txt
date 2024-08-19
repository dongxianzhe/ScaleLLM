include(cc_binary)

cc_binary(
  NAME
    test_callback_api
  SRCS
    test_callback_api.cu
  COPTS
    -I/usr/local/cuda/extras/CUPTI/include
  LINKOPTS
    -L/usr/local/cuda/extras/CUPTI/lib64
    -lcupti
)

cc_binary(
  NAME
    test_event_api
  SRCS
    test_event_api.cu
  COPTS
    -I/usr/local/cuda/extras/CUPTI/include
  LINKOPTS
    -L/usr/local/cuda/extras/CUPTI/lib64
    -lcupti
)