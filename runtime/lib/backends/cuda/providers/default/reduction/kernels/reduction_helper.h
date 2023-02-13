// Copyright (c) Megvii Inc.
// Licensed under the Apache License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/backends/cuda/device/common/dtype.h"
#include <cmath>
#include <cstdint>
#include <limits>

namespace brt {
namespace cuda {
namespace kernel {
namespace reduction {

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct SumOp {
  typedef wtype_ wtype;

  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) { return src[idx]; }
  __host__ __device__ void write(uint32_t idx, wtype val) { dst[idx] = val; }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
    return lhs + rhs;
  }
  __host__ __device__ SumOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MeanOp {
  typedef wtype_ wtype;

  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) { return src[idx]; }
  __host__ __device__ void write(uint32_t idx, wtype val) {
    dst[idx] = val / static_cast<wtype>(B);
  }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
    return lhs + rhs;
  }
  __host__ __device__ MeanOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct SumSqrOp {
  typedef wtype_ wtype;

  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) {
    return static_cast<wtype>(src[idx]) * static_cast<wtype>(src[idx]);
  }
  __host__ __device__ void write(uint32_t idx, wtype val) { dst[idx] = val; }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
    return lhs + rhs;
  }
  __host__ __device__ SumSqrOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct ProdOp {
  typedef wtype_ wtype;
  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) { return src[idx]; }
  __host__ __device__ void write(uint32_t idx, wtype val) { dst[idx] = val; }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
    return lhs * rhs;
  }
  __host__ __device__ ProdOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(1)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MinOp {
  typedef wtype_ wtype;
  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) { return src[idx]; }
  __host__ __device__ void write(uint32_t idx, wtype val) { dst[idx] = val; }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
    return lhs < rhs ? lhs : rhs;
  }
  __host__ __device__ MinOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(DTypeTraits<ctype_to_dtype<wtype>::value>::upper_bound())),
        src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype>
struct MinOp<src_ctype, dst_ctype, float> {
  typedef float wtype;
  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) { return src[idx]; }
  __host__ __device__ void write(uint32_t idx, wtype val) { dst[idx] = val; }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
#ifdef __CUDA_ARCH__
    return (isnan(lhs) || lhs < rhs) ? lhs : rhs;
#else
    return (std::isnan(lhs) || lhs < rhs) ? lhs : rhs;
#endif
  }
  __host__ __device__ MinOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(DTypeTraits<ctype_to_dtype<wtype>::value>::upper_bound())),
        src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MaxOp {
  typedef wtype_ wtype;
  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) { return src[idx]; }
  __host__ __device__ void write(uint32_t idx, wtype val) { dst[idx] = val; }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
    return lhs > rhs ? lhs : rhs;
  }
  __host__ __device__ MaxOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(DTypeTraits<ctype_to_dtype<wtype>::value>::lower_bound())),
        src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype>
struct MaxOp<src_ctype, dst_ctype, float> {
  typedef float wtype;
  const wtype INIT;

  src_ctype *src;
  dst_ctype *dst;
  const size_t B;

  __host__ __device__ wtype read(uint32_t idx) { return src[idx]; }
  __host__ __device__ void write(uint32_t idx, wtype val) { dst[idx] = val; }
  static __host__ __device__ wtype apply(wtype lhs, wtype rhs) {
#ifdef __CUDA_ARCH__
    return (isnan(lhs) || lhs > rhs) ? lhs : rhs;
#else
    return (std::isnan(lhs) || lhs > rhs) ? lhs : rhs;
#endif
  }
  __host__ __device__ MaxOp(src_ctype *src, dst_ctype *dst, size_t B)
      : INIT(wtype(DTypeTraits<ctype_to_dtype<wtype>::value>::lower_bound())),
        src(src), dst(dst), B(B) {}
};

} // namespace reduction
} // namespace kernel
} // namespace cuda
} // namespace brt
