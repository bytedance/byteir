//===- remove_control_flow.cc ---------------------------------*--- C++ -*-===//
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

#include <functional>
#include <queue>
#include <string>
#include <vector>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/remove_control_flow.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace llvm;

namespace {

struct RemoveTfSwitch : public OpRewritePattern<tf_executor::SwitchOp> {
  using OpRewritePattern<tf_executor::SwitchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tf_executor::SwitchOp op,
                                PatternRewriter &rewriter) const override {
    op.getFalseOutput().replaceAllUsesWith(op.getData());
    op.getTrueOutput().replaceAllUsesWith(op.getData());
    op->erase();
    return success();
  }
};

struct RemoveTfMerge : public OpRewritePattern<tf_executor::MergeOp> {
  using OpRewritePattern<tf_executor::MergeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tf_executor::MergeOp op,
                                PatternRewriter &rewriter) const override {
    Value operand0 = op.getOperand(0);
    Value operand1 = op.getOperand(1);
    auto switchOp = operand0.getDefiningOp<tf_executor::SwitchOp>();
    if (!switchOp)
      return failure();
    auto op1 = operand1.getDefiningOp<tf_executor::IslandOp>();
    if (!op1)
      return failure();
    if (!op1.WrapsSingleOp())
      return failure();
    auto sliceOp =
        llvm::dyn_cast_or_null<TF::StridedSliceOp>(&(op1.GetBody().front()));
    if (!sliceOp)
      return failure();

    op.getOutput().replaceAllUsesWith(operand0);
    op->erase();

    return success();
  }
};

struct RemoveControlFlowPass
    : public RemoveControlFlowBase<RemoveControlFlowPass> {
  void runOnOperation() override final {
    func::FuncOp graphOp = getOperation();
    MLIRContext *context = graphOp->getContext();
    RewritePatternSet patterns(context);
    patterns.add(std::make_unique<RemoveTfMerge>(context));
    patterns.add(std::make_unique<RemoveTfSwitch>(context));
    if (failed(applyPatternsAndFoldGreedily(graphOp, std::move(patterns)))) {
      signalPassFailure();
    }

    OpPassManager pm(graphOp.getOperationName());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());
    pm.addPass(createCanonicalizerPass());
    if (mlir::failed(runPipeline(pm, graphOp))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createRemoveControlFlowPass() {
  return std::make_unique<RemoveControlFlowPass>();
}
