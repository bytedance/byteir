//
// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved
// Licensed under the MIT license. See LICENSE.md file in the project root for
// full license information.
//
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/common.h"
#include <limits>

namespace brt {
namespace cuda {

struct fast_divmod {
  fast_divmod(int d = 1) {
    d_ = d == 0 ? 1 : d;
    BRT_ENFORCE(d_ >= 1 &&
                d_ <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

    for (l_ = 0; l_ < 32; l_++)
      if ((1U << l_) >= d_)
        break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
    M_ = static_cast<uint32_t>(m);
    // according to paper, the value of m' should fit in a unsigned integer.
    BRT_ENFORCE(M_ > 0 && M_ == m);
  }

  __device__ inline int div(int n) const {
#if defined(__CUDA_ARCH__)
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
#else
    // Using uint64_t for t, then t + n won't overflow.
    uint64_t t = ((uint64_t)M_ * n) >> 32;
    return static_cast<int>((t + n) >> l_);
#endif
  }

  __device__ inline int mod(int n) const { return n - div(n) * d_; }

  __device__ inline void divmod(int n, int &q, int &r) const {
    q = div(n);
    r = n - q * d_;
  }

  uint32_t d_; // divisor
  uint32_t M_; // m' in the paper.
  uint32_t l_; // l_ = ceil(log2(d_))
};

} // namespace cuda
} // namespace brt
