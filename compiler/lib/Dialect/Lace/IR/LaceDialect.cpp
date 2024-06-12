//===- LaceDialect.cpp ----------------------------------------------------===//
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

#include "byteir/Dialect/Lace/LaceDialect.h"

#include "byteir/Dialect/Ace/AceDialect.h" // ace types
#include "byteir/Utils/MemUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::lace;

#include "byteir/Dialect/Lace/LaceOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// lace dialect.
//===----------------------------------------------------------------------===//

void LaceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Lace/LaceOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
//  ReshapeOp
//===----------------------------------------------------------------------===//
LogicalResult ReshapeOp::verify() {
  // FIXME: only identify layout is supported now
  auto sourceMemRef = cast<MemRefType>(getSource().getType());
  auto targetMemRef = cast<MemRefType>(getTarget().getType());

  if (!sourceMemRef.getLayout().isIdentity() ||
      !targetMemRef.getLayout().isIdentity())
    return getOperation()->emitError()
           << "lace.reshape only supports identity layout";

  return success();
}

//===----------------------------------------------------------------------===//
//  SliceOp
//===----------------------------------------------------------------------===//
namespace {
/// Helpers to write more idiomatic operations.
namespace saturated_arith {
struct Wrapper {
  explicit Wrapper(int64_t v) : v(v) {}
  operator int64_t() { return v; }
  int64_t v;
};

Wrapper operator+(Wrapper a, int64_t b) {
  if (ShapedType::isDynamic(a) || ShapedType::isDynamic(b))
    return Wrapper(ShapedType::kDynamic);
  return Wrapper(a.v + b);
}

Wrapper operator*(Wrapper a, int64_t b) {
  if (ShapedType::isDynamic(a) || ShapedType::isDynamic(b))
    return Wrapper(ShapedType::kDynamic);
  return Wrapper(a.v * b);
}
} // namespace saturated_arith
} // namespace

LogicalResult SliceOp::verify() {
  // FIXME: only identify layout is supported now
  auto sourceMemRef = cast<MemRefType>(getSource().getType());
  auto targetMemRef = cast<MemRefType>(getTarget().getType());

  if (!sourceMemRef.getLayout().isIdentity() ||
      !targetMemRef.getLayout().isIdentity())
    return getOperation()->emitError()
           << "lace.slice only supports identity layout";

  // check whether target memref could be treated as a sub-memref on the source
  // memref
  SmallVector<int64_t> startIndices, limitIndices, strides;
  getValuesFromDenseIntElementsAttr(getStartIndices(), startIndices);
  getValuesFromDenseIntElementsAttr(getLimitIndices(), limitIndices);
  getValuesFromDenseIntElementsAttr(getStrides(), strides);

  if (!SliceOp::isValid(sourceMemRef, targetMemRef, startIndices, limitIndices,
                        strides))
    return getOperation()->emitError()
           << "Invalid memref type of lace.slice op";

  return success();
}

static MemRefType inferResultTypeOfSlice(MemRefType sourceMemRefType,
                                         ArrayRef<int64_t> startIndices,
                                         ArrayRef<int64_t> limitIndices,
                                         ArrayRef<int64_t> strides) {
  unsigned rank = sourceMemRefType.getRank();
  (void)rank;
  assert(startIndices.size() == rank && "startIndices length mismatch");
  assert(limitIndices.size() == rank && "limitIndices length mismatch");
  assert(strides.size() == rank && "strides length mismatch");

  // Extract source offset and strides.
  int64_t sourceOffset;
  SmallVector<int64_t, 4> sourceStrides;
  auto res = getStridesAndOffset(sourceMemRefType, sourceStrides, sourceOffset);
  assert(succeeded(res) && "SubViewOp expected strided memref type");
  (void)res;

  // Compute target offset whose value is:
  //   `sourceOffset + sum_i(startIndices_i * strides_i)`.
  int64_t targetOffset = sourceOffset;
  for (auto it : llvm::zip(startIndices, sourceStrides)) {
    auto startIndices = std::get<0>(it), targetStride = std::get<1>(it);
    using namespace saturated_arith;
    targetOffset = Wrapper(targetOffset) + Wrapper(startIndices) * targetStride;
  }

  // Compute target stride whose value is:
  //   `strides_i * staticStrides_i`.
  SmallVector<int64_t, 4> targetStrides;
  targetStrides.reserve(startIndices.size());
  for (auto it : llvm::zip(sourceStrides, strides)) {
    auto sourceStride = std::get<0>(it), staticStride = std::get<1>(it);
    using namespace saturated_arith;
    targetStrides.push_back(Wrapper(sourceStride) * staticStride);
  }

  SmallVector<int64_t, 4> sizes(rank);
  for (int64_t i = 0; i < rank; ++i) {
    sizes[i] = (limitIndices[i] - startIndices[i]) / strides[i];
  }

  // The type is now known.
  return MemRefType::get(
      sizes, sourceMemRefType.getElementType(),
      makeStridedLinearLayoutMap(targetStrides, targetOffset,
                                 sourceMemRefType.getContext()),
      sourceMemRefType.getMemorySpace());
}

int64_t lace::SliceOp::getOffsetElem() {
  auto sourceMemRef = cast<MemRefType>(getSource().getType());
  SmallVector<int64_t> startIndices, limitIndices, strides;
  getValuesFromDenseIntElementsAttr(getStartIndices(), startIndices);
  getValuesFromDenseIntElementsAttr(getLimitIndices(), limitIndices);
  getValuesFromDenseIntElementsAttr(getStrides(), strides);
  auto targetMemRef =
      inferResultTypeOfSlice(sourceMemRef, startIndices, limitIndices, strides);
  int64_t offset;
  SmallVector<int64_t> _;
  assert(succeeded(getStridesAndOffset(targetMemRef, _, offset)));
  return offset;
}

bool lace::SliceOp::isValid(MemRefType sourceMemRef,
                            MemRefType expectedTargetMemRef,
                            ArrayRef<int64_t> startIndices,
                            ArrayRef<int64_t> limitIndices,
                            ArrayRef<int64_t> strides) {

  auto inferredTargetMemRef =
      inferResultTypeOfSlice(sourceMemRef, startIndices, limitIndices, strides);

  if (!isStrided(sourceMemRef) || !isStrided(expectedTargetMemRef))
    return false;

  if (expectedTargetMemRef.getLayout().isIdentity()) {
    // FIXME: only identity layout is supported now
    if (isStaticShapeAndContiguousRowMajorEx(inferredTargetMemRef)) {
      if (static_cast<MemRefType>(
              MemRefType::Builder(inferredTargetMemRef).setLayout({})) ==
          expectedTargetMemRef) {
        return true;
      }
    }
  }

  return false;
}

#include "byteir/Dialect/Lace/LaceOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Lace/LaceOps.cpp.inc"
