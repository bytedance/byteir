//===- typecvt.h ----------------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "brt/core/framework/op_kernel_impl_base.h"
#include <immintrin.h>

namespace brt {
namespace cpu {

template <typename src_ctype, typename dst_ctype>
inline __attribute__((always_inline)) void
TypecvtKernelNaive(const src_ctype *__restrict src, dst_ctype *__restrict dst,
                   const size_t N) {
  for (size_t i = 0; i < N; ++i) {
    dst[i] = static_cast<dst_ctype>(src[i]);
  }
}

inline __attribute__((always_inline)) void
TypecvtKernelF32ToF16(const void *src_, void *dst_, const size_t N) {
  const float *src = reinterpret_cast<const float *>(src_);
  __m128i *dst = reinterpret_cast<__m128i *>(dst_);
  size_t i;
  for (i = 0; i < (N / 8) * 8; i += 8) {
    __m128i rst = _mm256_cvtps_ph(_mm256_loadu_ps(src), 0);
    _mm_storeu_si128(dst, rst);
    src += 8;
    dst++;
  }

  half_float::half *dst2 = reinterpret_cast<half_float::half *>(dst);
  for (; i < N; ++i) {
    *dst2 = static_cast<half_float::half>(*src);
    src++;
    dst2++;
  }
}

void TypecvtKernelF16ToF32(const void *src_, void *dst_, const size_t N) {
  const __m128i *src = reinterpret_cast<const __m128i *>(src_);
  float *dst = reinterpret_cast<float *>(dst_);
  size_t i;
  for (i = 0; i < (N / 8) * 8; i += 8) {
    __m256 rst = _mm256_cvtph_ps(_mm_loadu_si128(src));
    _mm256_storeu_ps(dst, rst);
    src++;
    dst += 8;
  }

  const half_float::half *src2 =
      reinterpret_cast<const half_float::half *>(src);
  for (; i < N; ++i) {
    *dst = static_cast<float>(*src2);
    src2++;
    dst++;
  }
}

template <DTypeEnum src_dtype, DTypeEnum dst_dtype> struct TypecvtImpl {
  TypecvtImpl(const OpAccessor &accessor) {
    N = accessor.GetNumElementsOfShape(accessor.GetArgShape(0));
  }

  common::Status Execute(const void *src, void *dst) {
    if constexpr (src_dtype == DTypeEnum::Float32 &&
                  dst_dtype == DTypeEnum::Float16) {
      TypecvtKernelF32ToF16(src, dst, N);
    } else if constexpr (src_dtype == DTypeEnum::Float16 &&
                         dst_dtype == DTypeEnum::Float32) {
      TypecvtKernelF16ToF32(src, dst, N);
    } else {
      TypecvtKernelNaive(
          reinterpret_cast<const typename DTypeTraits<src_dtype>::type_t *>(
              src),
          reinterpret_cast<typename DTypeTraits<dst_dtype>::type_t *>(dst), N);
    }
    return common::Status::OK();
  }

  size_t N;
};

template <DTypeEnum src_dtype, DTypeEnum dst_dtype>
using Typecvt =
    NaiveOpKernel<TypecvtImpl<src_dtype, dst_dtype>,
                  TypedOperand<const void *, 0>, TypedOperand<void *, 1>>;

} // namespace cpu
} // namespace brt
