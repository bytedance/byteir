//===- TrivialFusion.cpp --------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringMap.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

template <typename OpTy>
struct SingleOpPattern : public OpRewritePattern<OpTy> {
  SingleOpPattern(MLIRContext *context, const llvm::StringMap<StringRef> &lut)
      : OpRewritePattern<OpTy>(context), srcToName(lut) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    // avoid already fused
    if (op->template getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    MhloFusionPattern pattern;
    pattern.push_back(op);
    auto fusion = *createMhloFusionFromPattern(rewriter, pattern);
    // add attr
    fusion->setAttr(getByteIRTrivialFusionAttrName(),
                    UnitAttr::get(fusion.getContext()));

    // FIXME make this optional base on OpTy
    fusion->setAttr(byre::getByreForceComputeNameAttrName(),
                    UnitAttr::get(fusion.getContext()));

    auto found = srcToName.find(op.getOperation()->getName().getStringRef());
    if (found != srcToName.end()) {
      fusion->setAttr(byre::getByreComputeName(),
                      rewriter.getStringAttr(found->second));
    }

    return success();
  }

  const llvm::StringMap<StringRef> &srcToName;
};

struct TrivialFusionPass : public TrivialFusionBase<TrivialFusionPass> {

  TrivialFusionPass() : TrivialFusionBase() {
    // TODO: change to loading from outside
    // mhloNameMap.insert({"mhlo.rng_bit_generator", "RngBitGeneratorOp"});
    // mhloNameMap.insert({"mhlo.rng_normal", "RngNormalOp"});
    // mhloNameMap.insert({"mhlo.rng_uniform", "RngUniform"});
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateTrivialFusionPattern(patterns, mhloNameMap);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError(
          "TrivialFusionPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }

  llvm::StringMap<StringRef> mhloNameMap;
};
} // namespace

void mlir::populateTrivialFusionPattern(RewritePatternSet &patterns,
                                        llvm::StringMap<StringRef> &lut) {
  // FIXME: mhlo::RngNormalOp and RngUninformOp has been merged into RngOp.
  // patterns.add<SingleOpPattern<mhlo::RngBitGeneratorOp>,
  //              SingleOpPattern<mhlo::RngNormalOp>,
  //              SingleOpPattern<mhlo::RngUniformOp>>(patterns.getContext(),
  //              lut);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createTrivialFusionPass() {
  return std::make_unique<TrivialFusionPass>();
}
