//===- TestBroadcastDenseElementsAttr.cpp ---------------------------------===//
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

#include "byteir/Dialect/mhlo/Util/Util.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct BroadcastConstantPattern
    : public OpRewritePattern<mhlo::BroadcastInDimOp> {
  using OpRewritePattern<mhlo::BroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto constOp = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
    if (!constOp) {
      return failure();
    }
    auto broadcastDimensions =
        llvm::to_vector(op.getBroadcastDimensions().getValues<int64_t>());
    auto broadcastedAttr = createBroadcastedDenseElementsAttr(
        constOp.getValue().cast<DenseElementsAttr>(),
        op.getType().cast<ShapedType>(), broadcastDimensions);
    if (!broadcastedAttr.has_value()) {
      return failure();
    }
    auto newConstOp =
        rewriter.create<mhlo::ConstantOp>(op->getLoc(), *broadcastedAttr);
    rewriter.replaceOp(op, newConstOp.getOutput());
    return success();
  }
};

struct TestBroadcastDenseElementsAttrPass
    : public PassWrapper<TestBroadcastDenseElementsAttrPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestBroadcastDenseElementsAttrPass)

  StringRef getArgument() const final {
    return "test-broadcast-dense-elements-attr";
  }

  StringRef getDescription() const final {
    return "Test createBroadcastedDenseElementsAttrImpl()";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add(std::make_unique<BroadcastConstantPattern>(context));

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestBroadcastDenseElementsAttrPass() {
  PassRegistration<TestBroadcastDenseElementsAttrPass>();
}
} // namespace test
} // namespace byteir
