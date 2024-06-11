//===- reshape_movedown_string.cc -----------------------------*--- C++ -*-===//
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

#include "tf_mlir_ext/transforms/reshape_movedown_string.h"
#include "tf_mlir_ext/transforms/passes_detail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;

namespace {

struct ReshapeMovedownStringPattern : public OpRewritePattern<TF::EqualOp> {
  using OpRewritePattern<TF::EqualOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::EqualOp equal_op,
                                PatternRewriter &rewriter) const override {
    // convert 'string -> reshape -> equal' to 'string->equal->reshape'.
    // 1. match
    MLIRContext *context = equal_op->getContext();
    llvm::SmallVector<Type> types =
        llvm::to_vector(equal_op->getOperands().getTypes());
    if (types.size() != 2)
      return failure();
    for (Type &ty : types) {
      auto tensor_type = dyn_cast<mlir::TensorType>(ty);
      if (tensor_type) {
        if (!tensor_type.getElementType().isa<TF::StringType>()) {
          return failure();
        }
      }
    }
    auto reshapeOp = equal_op.getOperand(0).getDefiningOp<TF::ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto const_op = equal_op.getOperand(1).getDefiningOp<TF::ConstOp>();
    if (!const_op)
      return failure();

    // 2. rewrite
    Value input = reshapeOp.getOperand(0);
    Value shape = reshapeOp.getOperand(1);
    auto resultType = cast<RankedTensorType>(equal_op.getType());
    auto reshapeType = cast<RankedTensorType>(input.getType());
    if (!resultType.hasStaticShape() || !reshapeType.hasStaticShape())
      return failure();

    auto loc = UnknownLoc::get(context);
    Value equal_out = rewriter.create<TF::EqualOp>(
        loc, input, const_op, BoolAttr::get(context, true));
    Value reshape_out =
        rewriter.create<TF::ReshapeOp>(loc, resultType, equal_out, shape);
    rewriter.replaceOp(equal_op, reshape_out);
    return success();
  }
};

struct ReshapeMovedownStringPass
    : public ReshapeMovedownStringBase<ReshapeMovedownStringPass> {
  ReshapeMovedownStringPass() = default;

  void runOnOperation() override final {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.add(std::make_unique<ReshapeMovedownStringPattern>(ctx));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createReshapeMovedownStringPass() {
  return std::make_unique<ReshapeMovedownStringPass>();
}
