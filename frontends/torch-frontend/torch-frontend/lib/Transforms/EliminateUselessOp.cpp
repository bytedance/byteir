//===- EliminateUselessOp.cpp -----------------------------------*- C++ -*-===//
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

#include "torch-frontend/Transforms/EliminateUselessOp.h"
#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
struct EliminateTorchOperatorOpByPrefix : public OpRewritePattern<OperatorOp> {
  EliminateTorchOperatorOpByPrefix(MLIRContext *context, StringRef _prefix)
      : OpRewritePattern<OperatorOp>(context), prefix(_prefix) {}
  LogicalResult matchAndRewrite(OperatorOp op,
                                PatternRewriter &rewriter) const override {
    llvm::StringRef name = op.getNameAttr().getValue();
    if (name.starts_with(prefix) && op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "Expected op with name starts with." + prefix);
  }
  StringRef prefix;
};
} // namespace

namespace {
struct EliminateAtenWarnOp : public OpRewritePattern<AtenWarnOp> {
  using OpRewritePattern<AtenWarnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWarnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
struct EliminateUselessOpPass
    : public EliminateUselessOpBase<EliminateUselessOpPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    // Eliminate torch.profiler.xxx ops
    patterns.add<EliminateTorchOperatorOpByPrefix>(context, "profiler.");
    // Eliminate torch.aten.warn op
    patterns.add<EliminateAtenWarnOp>(context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createEliminateUselessOpPass() {
  return std::make_unique<EliminateUselessOpPass>();
}