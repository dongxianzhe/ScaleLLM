#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename float_t, size_t vec_size>
struct vec_t { // 设备上的向量，存在线程的寄存器中，长度为vec_size，类型为float_t
  FLASHINFER_INLINE float_t& operator[](size_t i); // 左值索引
  FLASHINFER_INLINE const float_t& operator[](size_t i) const; // 右值的索引
  FLASHINFER_INLINE void fill(float_t val); // val填充整个向量的每个元素
  FLASHINFER_INLINE void load(const float_t* ptr); // 从指针加载向量，这个指针指向的是global mem吗？
  FLASHINFER_INLINE void store(float_t* ptr) const; // 将向量写到指针所指
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src); // 从数据类型为T的向量转为float_t类型向量
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr); // 从指针所指加载向量，类型若不同则转换
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const; // 存到指针所指，类型不同先转换再store
  FLASHINFER_INLINE static void memcpy(float_t* dst, const float_t* src); // 
  FLASHINFER_INLINE float_t* ptr(); // 
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size> // 将src_float_t类型向量转换为tgt_float_t类型向量
FLASHINFER_INLINE void cast_from_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const vec_t<src_float_t, vec_size>& src) { // cast_from的具体实现，不同的vec_t都调用这个
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    dst[i] = tgt_float_t(src[i]); // ??? 这里不同数据类型的浮点数可以用等号吗？
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_load_impl(vec_t<tgt_float_t, vec_size>& dst,
                                      const src_float_t* src_ptr) { //cast_store的具体实现
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) { // 数据类型一样就直接加载
    dst.load(src_ptr);
  } else { // 数据类型不同先加载到tmp，然后在转换类型
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
FLASHINFER_INLINE void cast_store_impl(tgt_float_t* dst_ptr,
                                       const vec_t<src_float_t, vec_size>& src) {
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) { // 类型相同则直接store
    src.store(dst_ptr);
  } else { // 类型不同先转换到tmp向量，再store
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<half, 1> {
  half data;

  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val);
  FLASHINFER_INLINE void load(const half* ptr);
  FLASHINFER_INLINE void store(half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(half* dst, const half* src);
};

FLASHINFER_INLINE void vec_t<half, 1>::fill(half val) { data = val; }

FLASHINFER_INLINE void vec_t<half, 1>::load(const half* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<half, 1>::store(half* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<half, 1>::memcpy(half* dst, const half* src) { *dst = *src; }

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;

  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val);
  FLASHINFER_INLINE void load(const half* ptr);
  FLASHINFER_INLINE void store(half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(half* dst, const half* src);
};

FLASHINFER_INLINE void vec_t<half, 2>::fill(half val) { data = make_half2(val, val); }

FLASHINFER_INLINE void vec_t<half, 2>::load(const half* ptr) { data = *((half2*)ptr); }

FLASHINFER_INLINE void vec_t<half, 2>::store(half* ptr) const { *((half2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<half, 2>::memcpy(half* dst, const half* src) {
  *((half2*)dst) = *((half2*)src);
}

// half x 4

template <>
struct vec_t<half, 4> {
  uint2 data;

  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)(&data))[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val);
  FLASHINFER_INLINE void load(const half* ptr);
  FLASHINFER_INLINE void store(half* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(half* dst, const half* src);
};

FLASHINFER_INLINE void vec_t<half, 4>::fill(half val) {
  *(half2*)(&data.x) = make_half2(val, val);
  *(half2*)(&data.y) = make_half2(val, val);
}

FLASHINFER_INLINE void vec_t<half, 4>::load(const half* ptr) { data = *((uint2*)ptr); }

FLASHINFER_INLINE void vec_t<half, 4>::store(half* ptr) const { *((uint2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<half, 4>::memcpy(half* dst, const half* src) {
  *((uint2*)dst) = *((uint2*)src);
}

// half x 8 or more

template <size_t vec_size>
struct vec_t<half, vec_size> {
  uint4 data[vec_size / 8];
  FLASHINFER_INLINE half& operator[](size_t i) { return ((half*)data)[i]; }
  FLASHINFER_INLINE const half& operator[](size_t i) const { return ((const half*)data)[i]; }
  FLASHINFER_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  FLASHINFER_INLINE void fill(half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(half2*)(&(data[i].x)) = make_half2(val, val);
      *(half2*)(&(data[i].y)) = make_half2(val, val);
      *(half2*)(&(data[i].z)) = make_half2(val, val);
      *(half2*)(&(data[i].w)) = make_half2(val, val);
    }
  }
  FLASHINFER_INLINE void load(const half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(half* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(half* dst, const half* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }
};

/******************* vec_t<float> *******************/

// float x 1

template <>
struct vec_t<float, 1> {
  float data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 1>::fill(float val) { data = val; }

FLASHINFER_INLINE void vec_t<float, 1>::load(const float* ptr) { data = *ptr; }

FLASHINFER_INLINE void vec_t<float, 1>::store(float* ptr) const { *ptr = data; }

FLASHINFER_INLINE void vec_t<float, 1>::memcpy(float* dst, const float* src) { *dst = *src; }

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(&data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val);
  FLASHINFER_INLINE void load(const float* ptr);
  FLASHINFER_INLINE void store(float* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src);
};

FLASHINFER_INLINE void vec_t<float, 2>::fill(float val) { data = make_float2(val, val); }

FLASHINFER_INLINE void vec_t<float, 2>::load(const float* ptr) { data = *((float2*)ptr); }

FLASHINFER_INLINE void vec_t<float, 2>::store(float* ptr) const { *((float2*)ptr) = data; }

FLASHINFER_INLINE void vec_t<float, 2>::memcpy(float* dst, const float* src) {
  *((float2*)dst) = *((float2*)src);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  FLASHINFER_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  FLASHINFER_INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)dst)[i] = ((float4*)src)[i];
    }
  }
};