//===- DecomposeMhloCustomCallOps.cpp -------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/DecomposeMhloCustomCallOps.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {

struct DecomposeByteIRAddN : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != getAddNName())
      return failure();
    if (op.getOperands().size() < 2)
      return failure();

    Value result = rewriter.create<mhlo::AddOp>(op.getLoc(), op.getOperand(0),
                                                op.getOperand(1));
    for (size_t i = 2, e = op.getOperands().size(); i < e; i++) {
      result =
          rewriter.create<mhlo::AddOp>(op.getLoc(), result, op.getOperand(i));
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct DecomposeMhloCustomCallOpsPass
    : public DecomposeMhloCustomCallOpsBase<DecomposeMhloCustomCallOpsPass> {
  DecomposeMhloCustomCallOpsPass(ArrayRef<std::string> legalOps) {
    this->legalOps = legalOps;
  }

  void runOnOperation() override {
    legalOpsSet.clear();
    legalOpsSet.insert(legalOps.begin(), legalOps.end());

    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    if (!legalOpsSet.contains(getAddNName())) {
      patterns.add<DecomposeByteIRAddN>(context);
    }

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }

  llvm::StringSet<> legalOpsSet;
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createDecomposeMhloCustomCallOpsPass(ArrayRef<std::string> legalOps) {
  return std::make_unique<DecomposeMhloCustomCallOpsPass>(legalOps);
}
