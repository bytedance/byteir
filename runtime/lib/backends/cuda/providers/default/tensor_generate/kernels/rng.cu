//===- rng.cu -------------------------------------------------*--- C++ -*-===//
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

#include "./rng.h"
#include <atomic>
#include <curand_kernel.h>
#include <iostream>

// TODO: move to common header
#define DIVUP(x, y) (((x) + (y)-1) / (y))

namespace brt {
namespace cuda {
namespace kernel {
template <int NumElementsPerThread>
__global__ void _RngUniform(float *ptr, int32_t N, float base, float range,
                            size_t seed, size_t offset) {
  int32_t id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;

  curandState_t state;

  // initialize local state with 2^67 * sequence + offset steps
  curand_init(seed,   /* seed */
              id,     /* sequence */
              offset, /* offset */
              &state);

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      float value = curand_uniform(&state);
      ptr[id] = base + value * range;
      id += blockDim.x;
    }
  }
}

namespace details {
struct GlobalRngState {
public:
  GlobalRngState() : seed_(0) { offset_.store(0); }

  size_t next(size_t n) { return offset_.fetch_add(n); }
  size_t seed() { return seed_; }

  static GlobalRngState *inst() {
    static GlobalRngState _;
    return &_;
  }

private:
  size_t seed_;
  std::atomic<size_t> offset_;
};
} // namespace details

void RngUniform(cudaStream_t stream, float *ptr, size_t length, float low,
                float high) {
  constexpr int maxThreadsPerBlock = 256;
  constexpr int maxElementsPerThread = 4;
  int blocksPerGrid = static_cast<int>(
      DIVUP(length, maxThreadsPerBlock * maxElementsPerThread));
  int32_t N = static_cast<int32_t>(length);
  auto globalState = details::GlobalRngState::inst();
  size_t seed = globalState->seed();
  size_t offset = globalState->next(maxElementsPerThread);
  _RngUniform<maxElementsPerThread>
      <<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(
          ptr, N, low, high - low, seed, offset);
}

} // namespace kernel
} // namespace cuda
} // namespace brt