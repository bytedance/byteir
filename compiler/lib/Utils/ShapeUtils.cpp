//===- ShapeUtils.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace llvm;

namespace mlir {

LogicalResult reifyShapes(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications) {
  if (!op)
    return failure();

  if (op->hasTrait<hlo::OpTrait::CompatibleOperandsAndResultType>()) {
    // CompatibleOperandsAndResultType does not implement reify
    reifications.push_back(
        builder.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0)));
    return success();
  }

  // TODO: support nested function call
  if (auto origin = dyn_cast<InferShapedTypeOpInterface>(op)) {
    if (failed(origin.reifyReturnTypeShapes(builder, origin->getOperands(),
                                            reifications))) {
      return failure();
    }
  } else if (auto reifyFunc =
                 reifyReturnTypeShapes(op->getName().getStringRef())) {
    if (failed(reifyFunc(op, builder, op->getOperands(), reifications))) {
      return failure();
    }
  } else if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    auto inferFunc = reifyReturnTypeShapes(customCall.getCallTargetName());
    if (!inferFunc) {
      return failure();
    }
    if (failed(inferFunc(op, builder, op->getOperands(), reifications)))
      return failure();
  } else {
    // Return failure if op doesn't have InferShapedTypeOpInterface and not
    // registered.
    return failure();
  }

  return success();
}

FailureOr<SmallVector<Value>> createEmptyTensorForResult(OpBuilder &builder,
                                                         Operation *op) {
  SmallVector<Value> emptyTensors;
  bool resultsHasDynamicShape = false;
  for (auto &&result : op->getResults()) {
    if (auto resType = result.getType().template dyn_cast<ShapedType>()) {
      if (resType.hasStaticShape()) {
        auto emptyOp = builder.create<tensor::EmptyOp>(
            op->getLoc(), resType.getShape(), resType.getElementType());
        emptyTensors.emplace_back(emptyOp);
      } else {
        resultsHasDynamicShape = true;
        break;
      }
    }
  }

  if (resultsHasDynamicShape) {
    emptyTensors.clear();
    registerAllMhloReifyReturnTypeShapes();
    SmallVector<Value, 1> reifications;

    if (reifyShapes(builder, op, reifications).failed()) {
      return failure();
    }

    for (auto &&resultAndShape : llvm::zip(op->getResults(), reifications)) {
      SmallVector<Value, 1> dynamicSizes;
      auto resType = std::get<0>(resultAndShape).getType().cast<ShapedType>();
      for (size_t i = 0; i < resType.getRank(); ++i) {
        if (resType.isDynamicDim(i)) {
          auto dim = builder
                         .create<tensor::ExtractOp>(
                             op->getLoc(), std::get<1>(resultAndShape),
                             ValueRange{builder.create<arith::ConstantIndexOp>(
                                 op->getLoc(), static_cast<int64_t>(i))})
                         .getResult();
          dynamicSizes.emplace_back(dim);
        }
      }
      auto emptyOp =
          builder.create<tensor::EmptyOp>(op->getLoc(), resType, dynamicSizes);
      emptyTensors.emplace_back(emptyOp);
    }
  }
  return emptyTensors;
}

} // namespace mlir
