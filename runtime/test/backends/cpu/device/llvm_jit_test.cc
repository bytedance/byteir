//===- llvm_jit_test.cc ---------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cpu/device/llvm/jit.h"
#include "brt/core/common/status.h"
#include "brt/core/ir/engine_util.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <fstream>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::test;
using namespace brt::cpu;
using namespace std;

static std::string test_file_llvmjit = "test/test_files/add.ll";

namespace {
extern "C" {
void print() { std::cout << "testtesttest." << std::endl; }
}
} // namespace

TEST(LLVMJITTest, ADD) {
  auto llvmjit = LLVMJIT::Instance();
  ASSERT_TRUE(llvmjit->RegisterSymbol("print", reinterpret_cast<void *>(&print))
                  .IsOK());
  ASSERT_TRUE(llvmjit->LoadFromFile(test_file_llvmjit).IsOK());
  int length = 128;
  std::vector<int> a(length), b(length), c(length);
  {
    void *add_fn;
    ASSERT_TRUE(llvmjit->Lookup("add", &add_fn).IsOK());
    RandCPUBuffer(a.data(), length, 100);
    RandCPUBuffer(b.data(), length, 100);
    (*reinterpret_cast<void (*)(int *, int *, int *, int)>(add_fn))(
        a.data(), b.data(), c.data(), length);
    for (int i = 0; i < length; ++i) {
      ASSERT_EQ(a[i] + b[i], c[i]);
    }
  }
  {
    void *add_fn_packed;
    ASSERT_TRUE(llvmjit->LookupPacked("add", &add_fn_packed).IsOK());
    RandCPUBuffer(a.data(), length, 100);
    RandCPUBuffer(b.data(), length, 100);
    struct PackedArgs {
      void *a, *b, *c;
      int length;
    } args{a.data(), b.data(), c.data(), length};
    std::vector<void *> packed_args{&args.a, &args.b, &args.c, &args.length};
    (*reinterpret_cast<void (*)(void **)>(add_fn_packed))(packed_args.data());
    for (int i = 0; i < length; ++i) {
      ASSERT_EQ(a[i] + b[i], c[i]);
    }
  }
#if BRT_LLJIT_DEBUG
  // enable this to print optimized llvm module and dump compiled object to the
  // disk
  std::ofstream optimized(test_file_llvmjit + ".opt");
  ASSERT_TRUE(
      llvmjit->PrintOptimizedModule(test_file_llvmjit, optimized).IsOK());
  std::ofstream objs(test_file_llvmjit + ".o");
  ASSERT_TRUE(llvmjit->DumpObject(test_file_llvmjit, objs).IsOK());
#endif
}
