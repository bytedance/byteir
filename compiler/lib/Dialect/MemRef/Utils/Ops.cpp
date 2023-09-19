//===- Ops.cpp ------------------------------------------------------------===//
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

#include "byteir/Dialect/MemRef/Utils/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memref-utils-ops"
// #define llvm::outs() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;

namespace {

bool allStatic(ArrayRef<int64_t> array) {
  return llvm::all_of(array,
                      [](int64_t val) { return !ShapedType::isDynamic(val); });
}

} // namespace

// To check whether two subviews are overlapped or not, we can use a
// conservative approach. This means that when the function returns true, we can
// be certain that the two subviews are not overlapped. However, if the function
// returns false, we cannot say for certain whether the subviews are overlapped
// or not.
bool mlir::doSubViewsConservativelyNotOverlap(memref::SubViewOp lhs,
                                              memref::SubViewOp rhs) {
  Value lSrc = lhs.getSource();
  Value rSrc = rhs.getSource();
  // It is a conservative check, return false if the two subview ops don't share
  // the same source.
  if (lSrc != rSrc)
    return false;

  ArrayRef<int64_t> lOffsets = lhs.getStaticOffsets();
  ArrayRef<int64_t> lSizes = lhs.getStaticSizes();
  ArrayRef<int64_t> lStrides = lhs.getStaticStrides();
  ArrayRef<int64_t> rOffsets = rhs.getStaticOffsets();
  ArrayRef<int64_t> rSizes = rhs.getStaticSizes();
  ArrayRef<int64_t> rStrides = rhs.getStaticStrides();

  if (!(allStatic(lOffsets) && allStatic(lSizes) && allStatic(lStrides) &&
        allStatic(rOffsets) && allStatic(rSizes) && allStatic(rStrides)))
    return false;

  size_t rank = lhs.getStaticOffsets().size();
  if (rank == 0)
    return false;

  for (size_t idx = 0; idx < rank; ++idx) {
    DenseSet<int64_t> lhsPoints;
    for (int64_t i = 0; i < lSizes[idx]; ++i) {
      lhsPoints.insert(lOffsets[idx] + i * lStrides[idx]);
    }

    bool overlapped = false;
    for (int64_t i = 0; i < rSizes[idx]; ++i) {
      if (lhsPoints.contains(rOffsets[idx] + i * rStrides[idx])) {
        overlapped = true;
        break;
      }
    }
    if (!overlapped)
      return true;
  }

  return false;
}
