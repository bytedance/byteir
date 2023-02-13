//===- index_put.cu -------------------------------------------*--- C++ -*-===//
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

#include "./index_put.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include <algorithm>

namespace brt {
namespace cuda {
namespace kernel {

// a native_input_put putting entire inner_loop (feature dim) based outer_loop
// (embedding dim)
template <typename T, bool Accum>
__global__ void naive_index_put_kernel(T *inout, const int64_t *indices,
                                       const T *update,
                                       const int feature_bound) {
  int out_offset = indices[blockIdx.x];
  for (int idx = threadIdx.x; idx < feature_bound; idx += blockDim.x) {
    int in_idx = blockIdx.x * feature_bound + idx;
    int out_idx = out_offset * feature_bound + idx;
    T value = update[in_idx];
    if (Accum) {
      atomicAdd((T *)(inout + out_idx), value);
    } else {
      inout[out_idx] = value;
    }
  }
}

template <typename T, bool Accum>
void index_put(const T *input, const int64_t *indices, const T *update,
               T *output, const int index_count, const int feature_bound,
               const int size, cudaStream_t stream) {
  BRT_CUDA_CHECK(cudaMemcpyAsync(output, input, size * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream));
  dim3 grid = index_count;
  dim3 block = std::min(256, feature_bound);
  naive_index_put_kernel<T, Accum>
      <<<grid, block, 0, stream>>>(output, indices, update, feature_bound);
}

template void index_put<float, true>(const float *, const int64_t *,
                                     const float *, float *, const int,
                                     const int, const int, cudaStream_t);

} // namespace kernel
} // namespace cuda
} // namespace brt
