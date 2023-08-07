//===- ConvertRngToCustomCall.cpp -----------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/ConvertOpToCustomCall.h"

#include "./PassDetail.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct ConvertRngUniformToCustomCall : public OpRewritePattern<mhlo::RngOp> {
  using OpRewritePattern<mhlo::RngOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::RngOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRngDistribution() != mhlo::RngDistribution::UNIFORM) {
      return failure();
    }
    auto A = op.getA();
    auto B = op.getB();
    auto shape = op.getShape();
    TensorType resultType = op.getResult().getType();
    TensorType seedType = RankedTensorType::get({}, rewriter.getI64Type());
    auto getSeedOp =
        rewriter.create<byre::ComputeOp>(op->getLoc(), ArrayRef<Type>{seedType},
                                         "GetSeed", ValueRange(), ArrayAttr());
    auto getOffsetOp = rewriter.create<byre::ComputeOp>(
        op->getLoc(), ArrayRef<Type>{seedType}, "GetOffset", ValueRange(),
        ArrayAttr());
    SmallVector<Value> bufferArgs{A, B, getSeedOp.getResults()[0],
                                  getOffsetOp.getResults()[0]};
    if (!op.getType().hasStaticShape()) {
      bufferArgs.emplace_back(shape);
    }
    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), ArrayRef<Type>{resultType}, bufferArgs,
        getRngUniformName(), false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

struct ConvertOpToCustomCallPass
    : public ConvertOpToCustomCallBase<ConvertOpToCustomCallPass> {
public:
  ConvertOpToCustomCallPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<byre::ByreDialect>();
    registry.insert<mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateRngPatternToCustomCall(patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateRngPatternToCustomCall(RewritePatternSet &patterns) {
  patterns.add<ConvertRngUniformToCustomCall>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertOpToCustomCallPass() {
  return std::make_unique<ConvertOpToCustomCallPass>();
}
