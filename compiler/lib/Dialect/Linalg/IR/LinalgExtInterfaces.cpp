//===- LinalgExtInterfaces.cpp --------------------------------------------===//
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

#include "byteir/Dialect/Linalg/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::linalg_ext;

LogicalResult
mlir::linalg_ext::detail::verifyLinalgExtOpInterface(Operation *op) {
  LinalgExtOp linalgExtOp = cast<LinalgExtOp>(op);
  if (op->getNumResults()) {
    if (op->getNumResults() != linalgExtOp.getNumOutputs()) {
      return linalgExtOp.emitOpError(
          "expected number of outputs to be same as the number of results");
    }
    for (auto en : llvm::enumerate(op->getResultTypes())) {
      Type outputType = linalgExtOp.getOutputs()[en.index()].getType();
      if (en.value() != outputType) {
        return linalgExtOp.emitOpError("expected type of `outs` operand #")
               << en.index() << " " << outputType
               << " to be same as result type " << en.value();
      }
    }
  }
  return success();
}

#include "byteir/Dialect/Linalg/IR/LinalgExtOpInterfaces.cpp.inc" // IWYU pragma: export

template <typename Ty, typename DimOpTy>
static void getDimValues(OpBuilder &b, Location loc, Value v, Ty t,
                         SmallVector<OpFoldResult> &dimVals) {
  for (auto dim : llvm::enumerate(t.getShape())) {
    if (ShapedType::isDynamic(dim.value())) {
      Value dimVal = b.create<DimOpTy>(loc, v, dim.index());
      dimVals.push_back(dimVal);
    } else {
      Value constant = b.create<arith::ConstantIndexOp>(loc, dim.value());
      dimVals.push_back(constant);
    }
  }
}

LogicalResult mlir::linalg_ext::LinalgExtOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Operation *op = getOperation();
  for (auto output : getOutputs()) {
    SmallVector<OpFoldResult> dims;
    Type outputType = output.getType();
    if (auto rankedTensorType = dyn_cast<RankedTensorType>(outputType)) {
      getDimValues<RankedTensorType, tensor::DimOp>(b, op->getLoc(), output,
                                                    rankedTensorType, dims);
    } else if (auto memrefType = dyn_cast<MemRefType>(outputType)) {
      getDimValues<MemRefType, memref::DimOp>(b, op->getLoc(), output,
                                              memrefType, dims);
    } else if (!outputType.isIntOrIndexOrFloat()) {
      return op->emitOpError(
          "invalid type for output operand, expected tensor, "
          "memref or scalar type");
    }
    reifiedReturnShapes.push_back(std::move(dims));
  }
  return success();
}
