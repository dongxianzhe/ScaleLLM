#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)                     \
  [&]() -> bool {                                                                       \
    switch (pytorch_dtype) {                                                            \
      case at::ScalarType::Float:{                                                      \
        using c_type = float;                                                           \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case at::ScalarType::Half: {                                                      \
        using c_type = nv_half;                                                         \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      default:                                                                          \
        fprintf(stderr, " failed to dispatch data type ");                              \
        exit(1);                                                                        \
    }                                                                                   \
  }()

#define DISPATCH_group_size(expr, const_expr, ...)                                      \
  [&]() -> bool {                                                                       \
    switch (expr) {                                                                     \
      case 1: {                                                                         \
        constexpr int const_expr = 1;                                                   \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case 4: {                                                                         \
        constexpr int const_expr = 4;                                                   \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case 5: {                                                                         \
        constexpr int const_expr = 5;                                                   \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case 6: {                                                                         \
        constexpr int const_expr = 6;                                                   \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case 7: {                                                                         \
        constexpr int const_expr = 7;                                                   \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case 8: {                                                                         \
        constexpr int const_expr = 8;                                                   \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      default:                                                                          \
        fprintf(stderr, " failed to dispatch group_size %d", expr);                     \
        exit(1);                                                                        \
    }                                                                                   \
  }()

#define DISPATCH_head_dim(expr, const_expr, ...)                                        \
  [&]() -> bool {                                                                       \
    switch (expr) {                                                                     \
      case 64: {                                                                        \
        constexpr int const_expr = 64;                                                  \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case 128: {                                                                       \
        constexpr int const_expr = 128;                                                 \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      case 256: {                                                                       \
        constexpr int const_expr = 256;                                                 \
        return __VA_ARGS__();                                                           \
      }                                                                                 \
      default:                                                                          \
        fprintf(stderr, " failed to dispatch head_dim %d", expr);                       \
        exit(1);                                                                        \
    }                                                                                   \
  }()