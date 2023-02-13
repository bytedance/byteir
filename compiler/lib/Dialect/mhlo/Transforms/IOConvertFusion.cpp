//===- IOConvertFusion.cpp ------------------------------------*--- C++ -*-===//
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
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

// Note IOConvert will keep input/output sequence order as orginal op
template <typename OpTy>
struct IOConvertFusionPattern : public OpRewritePattern<OpTy> {
  IOConvertFusionPattern(MLIRContext *context, StringRef _byreComputeName)
      : OpRewritePattern<OpTy>(context), byreComputeName(_byreComputeName) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // early termination
    if (op->template getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    MhloFusionPattern pattern;
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;

    // handle input's convert
    for (unsigned idx = 0; idx < op->getNumOperands(); ++idx) {
      auto value = op->getOperand(idx);
      auto defOp = value.getDefiningOp();
      if (isa_and_nonnull<mhlo::ConvertOp>(defOp)) {
        auto cloned = replicateDefiningOp(rewriter, op, idx, 0);
        pattern.push_back(cloned);
        inputs.push_back(cloned->getOperand(0));
      } else if (isSplatMhloConstant(defOp)) {
        auto cloned = replicateDefiningOp(rewriter, op, idx, 0);
        pattern.push_back(cloned);
      } else {
        inputs.push_back(value);
      }
    }

    // op itself
    pattern.push_back(op);

    // handle output's convert
    for (unsigned idx = 0; idx < op->getNumResults(); ++idx) {
      auto value = op->getResult(idx);

      if (useCount(value) == 0) {
        continue;
      }

      if (useCount(value) > 1) {
        outputs.push_back(value);
        continue;
      }

      // UseCount(value) == 1
      auto user = *value.getUsers().begin();
      if (isa_and_nonnull<mhlo::ConvertOp>(user)) {
        pattern.push_back(user);
        outputs.push_back(user->getResult(0));
      } else {
        outputs.push_back(value);
      }
    }

    // note: single batch_norm_training should be fused
    // if (pattern.size() == 1) return failure();

    NamedAttrList attrs;
    // copy attrs to fusion op
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr(byreComputeName));
    for (const auto &attr : op->getAttrs()) {
      byre::appendByreComputeAttr(attrs, attr.getName().getValue(),
                                  attr.getValue());
    }

    mhlo::FusionOp fusionOp =
        createMhloFusionFromPattern(rewriter, inputs, outputs, pattern);
    fusionOp->setAttrs(attrs.getDictionary(this->getContext()));

    return success();
  }

  StringRef byreComputeName;
};

struct IOConvertFusionPass : public IOConvertFusionBase<IOConvertFusionPass> {
  IOConvertFusionPass() : IOConvertFusionBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateIOConvertBatchNormPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("IOConvertFusionPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateIOConvertBatchNormPattern(RewritePatternSet &patterns) {
  patterns.add<IOConvertFusionPattern<mhlo::BatchNormTrainingOp>>(
      patterns.getContext(), "BatchNormTrainingOp");
  patterns.add<IOConvertFusionPattern<mhlo::BatchNormGradOp>>(
      patterns.getContext(), "BatchNormGradOp");
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createIOConvertFusionPass() {
  return std::make_unique<IOConvertFusionPass>();
}