// Copyright (c) Megvii Inc.
// Licensed under the Apache License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include <cuda_runtime.h>

namespace brt {
namespace cuda {
namespace kernel {
template <typename wtype>
size_t get_reduce_workspace_in_bytes(size_t A, size_t B, size_t C);

template <typename T, typename Op>
void call_reduce(const T *input, T *output, size_t A, size_t B, size_t C,
                 void *, cudaStream_t stream);
} // namespace kernel
} // namespace cuda
} // namespace brt