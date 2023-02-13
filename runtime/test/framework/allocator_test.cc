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

#include "brt/core/framework/allocator.h"
#include "brt/core/framework/arena.h"
#include "brt/core/framework/bfc_arena.h"
#include "brt/test/common/config.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <string>

using namespace brt;
using namespace brt::test;

static void CheckResult(void *ptr, size_t size, char val) {
  CheckValues<char>((char *)ptr, size, val);
}

static inline void test_func(IAllocator *cpu_allocator) {
#if BRT_TEST_WITH_ASAN
  size_t large_size = 32 * 1024 * 1024;
#else
  size_t large_size = 1024 * 1024 * 1024;
#endif

  auto ptr = cpu_allocator->Alloc(large_size);
  EXPECT_TRUE(ptr != nullptr);
  // test the bytes are ok for read/write
  memset(ptr, -1, large_size);

  CheckResult(ptr, large_size, -1);
  cpu_allocator->Free(ptr);

  size_t small_size = 1024 * 1024; // 1MB
  size_t cnt = 1024;
  // check time for many allocation
  for (size_t s = 0; s < cnt; ++s) {
    void *raw = cpu_allocator->Alloc(small_size);
    cpu_allocator->Free(raw);
  }
  EXPECT_TRUE(true);
}

TEST(AllocatorTest, CPUBase) {
  CPUAllocator cpu_base_allocator;
  test_func(&cpu_base_allocator);
}

TEST(AllocatorTest, CPUArena) {
  BFCArena cpu_bfc_allocator(std::unique_ptr<IAllocator>(new CPUAllocator()),
                             1 << 30);
  test_func(&cpu_bfc_allocator);
}
