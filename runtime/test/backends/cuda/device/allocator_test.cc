//===- allocator_test.cc --------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/core/framework/arena.h"
#include "brt/core/framework/bfc_arena.h"
#include "brt/test/common/config.h"
#include "brt/test/common/cuda/util.h"
#include <cuda_runtime.h>
#include <string>

using namespace brt;
using namespace brt::test;

static void CheckResult(void *d_ptr, size_t size, char val) {
  CheckCUDABuffer<char>((char *)d_ptr, size, [&](char *h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      ASSERT_EQ(h_ptr[i], val);
    }
  });
}

static inline void test_func(IAllocator *cuda_allocator) {
#if BRT_TEST_WITH_ASAN
  size_t large_size = 32 * 1024 * 1024;
#else
  size_t large_size = 1024 * 1024 * 1024;
#endif

  auto ptr = cuda_allocator->Alloc(large_size);
  EXPECT_TRUE(ptr != nullptr);
  // test the bytes are ok for read/write
  cudaMemset(ptr, -1, large_size);
  CheckResult(ptr, large_size, -1);
  cuda_allocator->Free(ptr);

  size_t small_size = 1024 * 1024; // 1MB
  size_t cnt = 1024;
  // check time for many allocation
  for (size_t s = 0; s < cnt; ++s) {
    void *raw = cuda_allocator->Alloc(small_size);
    cuda_allocator->Free(raw);
  }
  EXPECT_TRUE(true);
}

TEST(AllocatorTest, CUDABase) {
  cudaSetDevice(0);
  CUDAAllocator cuda_base_alloc(0, "cuda"); // default CUDA
  test_func(&cuda_base_alloc);
}

TEST(AllocatorTest, CUDAArena) {
  cudaSetDevice(0);
  BFCArena cuda_arena_alloc(
      std::unique_ptr<IAllocator>(new CUDAAllocator(0, "cuda")), 1 << 30);
  test_func(&cuda_arena_alloc);
}
