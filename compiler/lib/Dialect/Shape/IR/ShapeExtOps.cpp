//===- ShapeExtOps.cpp ----------------------------------------------------===//
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

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>

using namespace mlir;

#include "byteir/Dialect/Shape/IR/ShapeExtOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// shape dialect.
//===----------------------------------------------------------------------===//

void mlir::shape_ext::ShapeExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Shape/IR/ShapeExtOps.cpp.inc"
      >();
}

namespace {

struct TieWithConst : public OpRewritePattern<shape_ext::TieOp> {
  using OpRewritePattern<shape_ext::TieOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape_ext::TieOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValue();
    RankedTensorType shapeType = cast<RankedTensorType>(value.getType());
    SmallVector<int64_t> shape = llvm::to_vector(shapeType.getShape());
    auto dims = op.getDims();
    SmallVector<Value> keepedDims;

    auto findNextDynamicDim = [&shape](auto it) {
      return std::find(it, shape.end(), ShapedType::kDynamic);
    };

    auto shpIt = findNextDynamicDim(shape.begin());
    for (auto dimIt = dims.begin(); dimIt != dims.end();
         dimIt++, shpIt = findNextDynamicDim(++shpIt)) {
      Value dimSize = *dimIt;
      Operation *defOp = dimSize.getDefiningOp();
      if (!defOp) {
        keepedDims.push_back(dimSize);
        continue;
      }

      IntegerAttr intAttr;
      if (matchPattern(dimSize, m_Constant(&intAttr))) {
        int64_t dimSizeInt = intAttr.getInt();
        *shpIt = dimSizeInt;
      } else {
        keepedDims.push_back(dimSize);
      }
    }

    if (keepedDims.size() == dims.size())
      return failure();
    value.setType(shapeType.clone(shape));
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    if (keepedDims.size() == 0) {
      op->erase();
    } else {
      op.getDimsMutable().assign(keepedDims);
    }

    funcOp.setType(FunctionType::get(
        funcOp.getContext(), funcOp.front().getArgumentTypes(),
        funcOp.front().getTerminator()->getOperandTypes()));

    return success();
  }
};

} // namespace

void mlir::shape_ext::TieOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<TieWithConst>(context);
}

LogicalResult mlir::shape_ext::TieOp::verify() {
  auto rankedTensorType = dyn_cast<RankedTensorType>(getValue().getType());
  if (!rankedTensorType)
    return emitError() << "The value's type should be RankedTensorType";
  auto numDynShape =
      llvm::count_if(rankedTensorType.getShape(), [](int64_t dimSize) {
        return dimSize == ShapedType::kDynamic;
      });
  if (size_t(numDynShape) != getDims().size())
    return emitError() << "The number of tie's dims and the dynamic size of "
                          "the original value don't match.";

  return success();
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Shape/IR/ShapeExtOps.cpp.inc"
