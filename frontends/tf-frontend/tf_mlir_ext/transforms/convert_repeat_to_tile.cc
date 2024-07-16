//===- convert_repeat_to_tile.cc ------------------------------*--- C++ -*-===//
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

#include "tf_mlir_ext/transforms/convert_repeat_to_tile.h"
#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/utils/utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace mlir::tfext;
using namespace llvm;

namespace {

struct ConvertRepeatToTilePattern : public RewritePattern {
  ConvertRepeatToTilePattern(MLIRContext *context, PatternBenefit benefits = 1)
      : RewritePattern("tf.Repeat", benefits, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    assert(op->getName().getStringRef() == "tf.Repeat");
    auto repeatType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!repeatType)
      return failure();

    auto input = op->getOperand(0);
    auto inputType = dyn_cast<ShapedType>(input.getType());
    if (!inputType)
      return failure();

    auto times = op->getOperand(1);
    auto timesType = dyn_cast<ShapedType>(times.getType());
    if (!timesType || !timesType.hasStaticShape())
      return failure();

    if (inputType.getRank() != 2 || timesType.getRank() != 1) {
      return failure();
    }

    if (!times.getDefiningOp())
      return failure();
    auto timesConstOp = times.getDefiningOp<TF::ConstOp>();
    if (!timesConstOp)
      return failure();

    auto constAttr = timesConstOp.getValue();
    if (!constAttr.isSplat())
      return failure();
    if (!constAttr.getElementType().isInteger(64))
      return failure();

    int64_t multiplyer = constAttr.getSplatValue<int64_t>();

    llvm::SmallVector<int64_t> tileConstArray = {1, multiplyer};
    auto tileConstType =
        mlir::RankedTensorType::get({2}, constAttr.getElementType());
    auto tileConstAttr = mlir::DenseElementsAttr::get(
        tileConstType, llvm::ArrayRef(tileConstArray));
    auto tileConstOp =
        rewriter.create<TF::ConstOp>(op->getLoc(), tileConstAttr);
    llvm::SmallVector<int64_t> tileShape = {
        timesType.getShape()[0], inputType.getShape()[1] * multiplyer};
    auto tileType = inputType.clone(tileShape);
    auto tileOp = rewriter.create<TF::TileOp>(op->getLoc(), tileType, input,
                                              tileConstOp.getOutput());

    llvm::SmallVector<int64_t> reshapeShape;
    for (auto s : inputType.getShape()) {
      reshapeShape.push_back(s);
    }
    reshapeShape[0] *= multiplyer;
    auto reshapeConstType =
        mlir::RankedTensorType::get({2}, constAttr.getElementType());
    auto reshapeConstAttr = mlir::DenseElementsAttr::get(
        reshapeConstType, llvm::ArrayRef(reshapeShape));
    auto reshapeConstOp =
        rewriter.create<TF::ConstOp>(op->getLoc(), reshapeConstAttr);

    auto reshapeType =
        mlir::RankedTensorType::get(reshapeShape, inputType.getElementType());
    auto reshapeOp = rewriter.create<TF::ReshapeOp>(op->getLoc(), reshapeType,
                                                    tileOp.getOutput(),
                                                    reshapeConstOp.getOutput());
    rewriter.replaceOp(op, reshapeOp.getOperation());

    return success();
  }
};

struct ConvertRepeatToTilePass
    : public ConvertRepeatToTileBase<ConvertRepeatToTilePass> {
  ConvertRepeatToTilePass() = default;

  void runOnOperation() override final {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(ctx);

    patterns.add(std::make_unique<ConvertRepeatToTilePattern>(ctx));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createConvertRepeatToTilePass() {
  return std::make_unique<ConvertRepeatToTilePass>();
}
