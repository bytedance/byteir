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
#include "half/half.hpp"
#include "gtest/gtest.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <immintrin.h>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::test;
using namespace brt::cpu;
using namespace std;

static std::string test_file_add = "test/test_files/LLJIT/add.ll";
static std::string test_file_typecvt = "test/test_files/LLJIT/typecvt.ll";

namespace {
extern "C" {
void print() { std::cout << "testtesttest." << std::endl; }
}

inline __attribute__((always_inline)) void
TypecvtKernelF32ToF16(const void *src_, void *dst_, const size_t N) {
  const float *src = reinterpret_cast<const float *>(src_);
  __m128i *dst = reinterpret_cast<__m128i *>(dst_);
  size_t i;
  for (i = 0; i < (N / 8) * 8; i += 8) {
    __m128i rst = _mm256_cvtps_ph(_mm256_loadu_ps(src), 0);
    _mm_storeu_si128(dst, rst);
    src += 8;
    dst++;
  }

  half_float::half *dst2 = reinterpret_cast<half_float::half *>(dst);
  for (; i < N; ++i) {
    *dst2 = static_cast<half_float::half>(*src);
    src++;
    dst2++;
  }
}
} // namespace

TEST(LLVMJITTest, ADD) {
  auto llvmjit = LLVMJIT::Create();
  ASSERT_TRUE(llvmjit->RegisterSymbol("print", reinterpret_cast<void *>(&print))
                  .IsOK());
  ASSERT_TRUE(llvmjit->LoadFromFile(test_file_add).IsOK());
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
  std::ofstream optimized(test_file_add + ".opt");
  ASSERT_TRUE(llvmjit->PrintOptimizedModule(test_file_add, optimized).IsOK());
  std::ofstream objs(test_file_add + ".o");
  ASSERT_TRUE(llvmjit->DumpObject(test_file_add, objs).IsOK());
#endif
}

TEST(LLVMJITTest, TypeCvt) {
  auto llvmjit = LLVMJIT::Create();
  ASSERT_TRUE(llvmjit->LoadFromFile(test_file_typecvt).IsOK());
  std::vector<int64_t> shape{1, 224, 224, 3};
  int length = 224 * 224 * 3;
  std::vector<float> input_buf(length);
  std::vector<half_float::half> output_buf(length);
  {
    void *fn;
    ASSERT_TRUE(llvmjit->Lookup("_mlir_ciface_Unknown0", &fn).IsOK());
    RandCPUBuffer(input_buf.data(), length);
    MLIREngineMemRefDescriptor input(input_buf.data(), shape),
        output(output_buf.data(), shape);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10000; ++i) {
      // TypecvtKernelF32ToF16(input_buf.data(), output_buf.data(), length);
      (*reinterpret_cast<void (*)(void *, void *)>(fn))(input.GetMemrefPtr(),
                                                        output.GetMemrefPtr());
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Duration :"
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                     .count()
              << std::endl;
    for (int i = 0; i < length; ++i) {
      ASSERT_NEAR(static_cast<half_float::half>(input_buf[i]), output_buf[i],
                  1e-6);
    }
  }
}