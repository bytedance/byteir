//===- elementwise.cu -----------------------------------------*--- C++ -*-===//
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

namespace brt {
namespace cuda {
namespace kernel {

template <typename T>
__global__ void add_kernel(const T *input_1, const T *input_2, T *output,
                           int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    output[idx] = input_1[idx] + input_2[idx];
  }
}

// instantiate
template __global__ void add_kernel<float>(const float *, const float *,
                                           float *, int);
template __global__ void add_kernel<int>(const int *, const int *, int *, int);

} // namespace kernel
} // namespace cuda
} // namespace brt
