//===- init_symbols.cc ----------------------------------------*--- C++ -*-===//
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

#include "./init_symbols.h"
#include "brt/backends/cpu/device/llvm/jit.h"
#include "brt/core/ir/engine_util.h"

using namespace brt;

namespace {
extern "C" void memrefCopy(int64_t elemSize,
                           MLIRUnrankedMemRefType<char> *srcArg,
                           MLIRUnrankedMemRefType<char> *dstArg) {

  MLIRDynamicMemRefType<char> src(*srcArg);
  MLIRDynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0)
      return;

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    // Copy over the element, byte by byte.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}
} // namespace

namespace brt {
namespace cpu {

void InitJITKernelRTSymbols(LLVMJIT *jit) {
#define REG2(name, symbol)                                                     \
  if (!jit->Lookup(name, nullptr).IsOK()) {                                    \
    BRT_ENFORCE(                                                               \
        jit->RegisterSymbol(name, reinterpret_cast<void *>(&symbol)).IsOK());  \
  }
#define REG(symbol) REG2(#symbol, symbol)

  REG(memrefCopy);
  // TODO: replace with the call of session host allocator's corresponding
  // method
  REG2("malloc", ::malloc);
  REG2("free", ::free);

#undef REG
#undef REG2
} // namespace cpu

} // namespace cpu
} // namespace brt
