//===- MatmulLayoutTransform.cpp ------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/MatmulLayoutTransform.h"
#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

// convert DotOp to DotGeneralOp by target layout
LogicalResult tryRewrite(mhlo::DotOp op, OpBuilder builder,
                         bool transposeConstantOnly, std::string targetLayout) {
  assert(targetLayout.size() == 3);
  // dot is a rrr matmul
  std::string defaultLayout = "rrr";
  if (targetLayout == defaultLayout)
    return failure();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  if (transposeConstantOnly) {
    if (targetLayout[0] != defaultLayout[0] &&
        !lhs.getDefiningOp<mhlo::ConstantOp>())
      return failure();
    if (targetLayout[1] != defaultLayout[1] &&
        !rhs.getDefiningOp<mhlo::ConstantOp>())
      return failure();
  }
  int64_t lhsContractingDimension = 1;
  int64_t rhsContractingDimension = 0;
  if (targetLayout[0] != defaultLayout[0]) {
    auto lhsType = lhs.getType().cast<RankedTensorType>();
    assert(lhsType.getRank() == 2);
    auto shape = lhsType.getShape();
    RankedTensorType newType =
        RankedTensorType::get({shape[1], shape[0]}, lhsType.getElementType());
    builder.setInsertionPoint(op);
    auto trans_op = builder.create<mhlo::TransposeOp>(
        op.getLoc(), newType, lhs, builder.getI64TensorAttr({1, 0}));
    lhsContractingDimension = 0;
    lhs = trans_op.getResult();
  }
  if (targetLayout[1] != defaultLayout[1]) {
    auto rhsType = rhs.getType().cast<RankedTensorType>();
    assert(rhsType.getRank() == 2);
    auto shape = rhsType.getShape();
    RankedTensorType newType =
        RankedTensorType::get({shape[1], shape[0]}, rhsType.getElementType());
    builder.setInsertionPoint(op);
    auto trans_op = builder.create<mhlo::TransposeOp>(
        op.getLoc(), newType, rhs, builder.getI64TensorAttr({1, 0}));
    rhsContractingDimension = 1;
    rhs = trans_op.getResult();
  }
  builder.setInsertionPoint(op);
  if (targetLayout[2] == defaultLayout[2]) {
    auto dotNums = mhlo::DotDimensionNumbersAttr::get(
        op.getContext(), {}, {}, {lhsContractingDimension},
        {rhsContractingDimension});
    auto dotOp = builder.create<mhlo::DotGeneralOp>(
        op.getLoc(), op.getType(), lhs, rhs, dotNums,
        op.getPrecisionConfig().value_or(nullptr));
    op.getResult().replaceAllUsesWith(dotOp);
    op.erase();
  } else {
    mlir::StringAttr config;
    if (targetLayout[0] == 'r' && targetLayout[1] == 'r')
      config = mlir::StringAttr::get(builder.getContext(), "mk,kn->nm");
    if (targetLayout[0] == 'r' && targetLayout[1] == 'c')
      config = mlir::StringAttr::get(builder.getContext(), "mk,nk->nm");
    if (targetLayout[0] == 'c' && targetLayout[1] == 'r')
      config = mlir::StringAttr::get(builder.getContext(), "km,kn->nm");
    if (targetLayout[0] == 'c' && targetLayout[1] == 'c')
      config = mlir::StringAttr::get(builder.getContext(), "km,nk->nm");
    auto outType = op.getType().cast<RankedTensorType>();
    auto outShape = outType.getShape();
    auto transposeType = RankedTensorType::get({outShape[1], outShape[0]},
                                               outType.getElementType());
    auto EinsumOp = builder.create<mhlo::EinsumOp>(op.getLoc(), transposeType,
                                                   lhs, rhs, config);
    auto TransposeOp = builder.create<mhlo::TransposeOp>(
        EinsumOp.getLoc(), op.getType(), EinsumOp.getResult(),
        builder.getI64TensorAttr({1, 0}));
    op.getResult().replaceAllUsesWith(TransposeOp);
    op.erase();
  }
  return success();
}

struct MatmulLayoutTransformPass
    : public MatmulLayoutTransformBase<MatmulLayoutTransformPass> {

  explicit MatmulLayoutTransformPass(bool transposeConstantOnly,
                                     std::string targetLayout)
      : MatmulLayoutTransformBase<MatmulLayoutTransformPass>() {
    this->transposeConstantOnly = transposeConstantOnly;
    this->targetLayout = targetLayout;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);
    funcOp.walk([&](mhlo::DotOp op) {
      (void)tryRewrite(op, builder, transposeConstantOnly, targetLayout);
    });

    RewritePatternSet patterns(context);
    mhlo::getCanonicalizationExtPatterns(patterns, context);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createMatmulLayoutTransformPass(bool transposeConstantOnly,
                                      std::string targetLayout) {
  return std::make_unique<MatmulLayoutTransformPass>(transposeConstantOnly,
                                                     targetLayout);
}
