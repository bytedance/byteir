//===- fill.cu ------------------------------------------------*--- C++ -*-===//
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

#include "./fill.h"

// TODO: move to common header
#define DIVUP(x, y) (((x) + (y)-1) / (y))

namespace brt {
namespace cuda {
namespace kernel {
template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Fill(T *output_data, T val, int32_t N) {
  int32_t id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = val;
      id += blockDim.x;
    }
  }
}

template <typename T>
void Fill(cudaStream_t stream, T *output, T value, size_t count) {
  constexpr int maxThreadsPerBlock = 256;
  constexpr int maxElementsPerThread = 4;
  int blocksPerGrid =
      static_cast<int>(DIVUP(count, maxThreadsPerBlock * maxElementsPerThread));
  int32_t N = static_cast<int32_t>(count);
  _Fill<T, maxThreadsPerBlock, maxElementsPerThread>
      <<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(output, value, N);
}

#define INST(T) template void Fill<T>(cudaStream_t, T *, T, size_t);

INST(float)
INST(int64_t)
INST(double)
INST(__half)
INST(int8_t)

#undef INST

} // namespace kernel
} // namespace cuda
} // namespace brt