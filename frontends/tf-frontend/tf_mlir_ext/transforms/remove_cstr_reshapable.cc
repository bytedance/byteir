//===- remove_cstr_reshapable.cc ------------------------------*--- C++ -*-===//
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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/remove_cstr_reshapable.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "remove_cstr_reshapable"

namespace {
/// Removal patterns.
class RemoveCstrReshapableOp : public OpRewritePattern<mhlo::CstrReshapableOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CstrReshapableOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op.getOperation(), true);
    return success();
  }
};

/// Removal pass.
class RemoveCstrReshapablePass
    : public RemoveCstrReshapableBase<RemoveCstrReshapablePass> {

  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    patterns.add<RemoveCstrReshapableOp>(patterns.getContext());

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createRemoveCstrReshapablePass() {
  return std::make_unique<RemoveCstrReshapablePass>();
}
