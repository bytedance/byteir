//===- transpose.cu -------------------------------------------*--- C++ -*-===//
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

#include <cuda_fp16.h>

namespace brt {
namespace cuda {
namespace kernel {

template <typename T>
__global__ void transpose_naive_2d_kernel(const T *input, T *output, int m,
                                          int n) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (iy < m && ix < n) {
    int in_idx = iy * n + ix;
    int out_idx = ix * m + iy;
    output[out_idx] = input[in_idx];
  }
}

template <typename T>
void transpose_naive_2d(const T *input, T *output, int m, int n, dim3 grid,
                        dim3 block, cudaStream_t stream) {
  transpose_naive_2d_kernel<T><<<grid, block, 0, stream>>>(input, output, m, n);
}

// instantiate
template void transpose_naive_2d<float>(const float *, float *, int, int, dim3,
                                        dim3, cudaStream_t);
template void transpose_naive_2d<__half>(const __half *, __half *, int, int,
                                         dim3, dim3, cudaStream_t);

} // namespace kernel
} // namespace cuda
} // namespace brt
