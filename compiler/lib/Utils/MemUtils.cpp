//===- MemUtils.cpp -------------------------------------------------------===//
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

#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace llvm;
using namespace mlir;

Attribute mlir::wrapIntegerMemorySpace(unsigned space, MLIRContext *ctx) {
  if (space == 0)
    return nullptr;
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

std::optional<int64_t> mlir::getRank(Value val) {
  if (auto shapedType = dyn_cast<ShapedType>(val.getType())) {
    return shapedType.getRank();
  }
  return std::nullopt;
}

std::optional<Value> mlir::getDimSize(OpBuilder &b, Value val, unsigned idx) {
  if (auto shapedType = dyn_cast<ShapedType>(val.getType())) {
    auto loc = val.getLoc();
    if (shapedType.isDynamicDim(idx)) {
      auto dimOp = b.create<memref::DimOp>(loc, val, idx);
      return dimOp.getResult();
    } else {
      auto cOp =
          b.create<arith::ConstantIndexOp>(loc, shapedType.getDimSize(idx));
      return cOp.getResult();
    }
  }
  return std::nullopt;
}

// Create an alloc based on an existing Value 'val', with a given space.
// return std::nullopt, if not applicable.
std::optional<Value> mlir::createAlloc(OpBuilder &b, Value val,
                                       unsigned space) {
  // early termination if not a memref
  if (!isa<MemRefType>(val.getType()))
    return std::nullopt;

  auto oldMemRefType = cast<MemRefType>(val.getType());

  auto spaceAttr = wrapIntegerMemorySpace(space, b.getContext());

  SmallVector<Value, 4> dynValue;

  auto shape = oldMemRefType.getShape();

  auto newMemRefType = MemRefType::get(shape, oldMemRefType.getElementType(),
                                       nullptr /*layout*/, spaceAttr);

  for (unsigned idx = 0, n = shape.size(); idx < n; ++idx) {
    if (shape[idx] == ShapedType::kDynamic) {
      auto maybeValue = getDimSize(b, val, idx);
      if (!maybeValue.has_value()) {
        return std::nullopt;
      }

      dynValue.push_back(*maybeValue);
    }
  }

  auto loc = val.getLoc();
  auto alloc = b.create<memref::AllocOp>(loc, newMemRefType, dynValue);
  return alloc.getResult();
}

// Get byte shift from the original allocation operation or function argument.
// Note that `shift` is different from `offset`, since `shift` is used for
// contiguous memory, while `offset` is used in multi-dimenstional situation.
// return std::nullopt, if val is not of type MemRefType or it could not be
// determined.
std::optional<int64_t> mlir::getByteShiftFromAllocOrArgument(Value val) {
  auto memRefType = dyn_cast_or_null<MemRefType>(val.getType());
  if (!memRefType)
    return std::nullopt;
  Operation *op = val.getDefiningOp();
  if (!op || isa<memref::AllocOp>(op)) {
    return 0;
  } else if (auto viewOp = dyn_cast<memref::ViewOp>(op)) {
    Value offsetVal = viewOp.getByteShift();
    if (auto offsetOp = offsetVal.getDefiningOp<arith::ConstantOp>()) {
      if (auto offsetLit = dyn_cast_or_null<IntegerAttr>(offsetOp.getValue())) {
        int64_t curOffset = offsetLit.getInt();
        std::optional<int64_t> subOffset =
            getByteShiftFromAllocOrArgument(viewOp.getSource());
        if (!subOffset.has_value())
          // the byte shift of viewOp's source is None
          return std::nullopt;
        else
          return curOffset + *subOffset;
      } else {
        llvm_unreachable(
            "view op's byte shift is arith.constant but not of Integer type.");
      }
    } else {
      // the byte shift of view op is not arith.constant
      return std::nullopt;
    }
  } else if (auto subViewOp = dyn_cast<memref::SubViewOp>(op)) {
    return getByteShiftFromAllocOrArgument(subViewOp.getSource());
  } else if (auto viewLike = dyn_cast<ViewLikeOpInterface>(op)) {
    return getByteShiftFromAllocOrArgument(viewLike.getViewSource());
  }
  return std::nullopt;
}

bool mlir::isStatic(MemRefType t) {
  ShapedType shape = dyn_cast_or_null<ShapedType>(t);
  if (!shape)
    return false;
  if (!shape.hasStaticShape())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(t, strides, offset)))
    return false;
  if (offset == ShapedType::kDynamic)
    return false;
  for (auto stride : strides) {
    if (stride == ShapedType::kDynamic)
      return false;
  }
  return true;
}

std::optional<int64_t> mlir::getSizeInBits(MemRefType t) {
  if (!isStatic(t))
    return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset;
  assert(succeeded(getStridesAndOffset(t, strides, offset)));
  int64_t numElems = offset;
  ArrayRef<int64_t> shapes = cast<ShapedType>(t).getShape();
  for (auto strideAndShape : zip(strides, shapes)) {
    int64_t stride = std::get<0>(strideAndShape);
    int64_t shape = std::get<1>(strideAndShape);
    numElems = std::max(numElems, offset + stride * shape);
  }

  auto elementType = t.getElementType();
  if (elementType.isIntOrFloat())
    return elementType.getIntOrFloatBitWidth() * numElems;

  if (auto complexType = dyn_cast<ComplexType>(elementType)) {
    elementType = complexType.getElementType();
    return elementType.getIntOrFloatBitWidth() * numElems * 2;
  }
  return std::nullopt;
}

MemRefType mlir::cloneMemRefTypeWithMemSpace(MemRefType t, Attribute space) {
  return MemRefType::get(t.getShape(), t.getElementType(), t.getLayout(),
                         space);
}

MemRefType mlir::cloneMemRefTypeAndRemoveMemSpace(MemRefType t) {
  return MemRefType::get(t.getShape(), t.getElementType(), t.getLayout());
}

bool mlir::isStaticShapeAndContiguousRowMajorEx(MemRefType memref) {
  auto isStaticShapeAndContiguousRowMajor = [](MemRefType type) {
    if (!type.hasStaticShape())
      return false;

    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(getStridesAndOffset(type, strides, offset)))
      return false;

    int64_t runningStride = 1;
    for (unsigned i = strides.size(); i > 0; --i) {
      if (strides[i - 1] != runningStride)
        return false;
      runningStride *= type.getDimSize(i - 1);
    }
    return true;
  };

  auto canonicalizer = [](MemRefType memref) -> MemRefType {
    if (!memref.hasStaticShape())
      return memref;
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(getStridesAndOffset(memref, strides, offset)))
      return memref;
    int64_t runningSize = 1;
    for (auto &&pi :
         llvm::zip(llvm::reverse(strides), llvm::reverse(memref.getShape()))) {
      auto &&stride = std::get<0>(pi);
      auto &&size = std::get<1>(pi);
      if (stride != ShapedType::kDynamic && size == 1) {
        stride = runningSize;
      }
      runningSize *= size;
    }
    AffineMap newLayout =
        makeStridedLinearLayoutMap(strides, offset, memref.getContext());
    return MemRefType::get(memref.getShape(), memref.getElementType(),
                           newLayout);
  };
  return isStaticShapeAndContiguousRowMajor(canonicalizer(memref));
}
