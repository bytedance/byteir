//===- ConvertHloToCat.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/HloToCat/ConvertHloToCat.h"
#include "byteir/Dialect/Cat/IR/CatDialect.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"

#include "../PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {
bool isNormalizedGemm(mhlo::DotGeneralOp op, const std::string &layout) {
  auto dotDimensionNumbers = op.getDotDimensionNumbers();
  if (dotDimensionNumbers.getLhsBatchingDimensions().size() != 0) {
    return false;
  }
  if (dotDimensionNumbers.getRhsBatchingDimensions().size() != 0) {
    return false;
  }
  if (dotDimensionNumbers.getLhsContractingDimensions().size() != 1) {
    return false;
  }
  if (dotDimensionNumbers.getRhsContractingDimensions().size() != 1) {
    return false;
  }
  auto lhsRank = cast<ShapedType>(op.getLhs().getType()).getRank();
  auto rhsRank = cast<ShapedType>(op.getRhs().getType()).getRank();
  if (lhsRank != 2 || rhsRank != 2) {
    return false;
  }

  int64_t lhsContractingDimension =
      dotDimensionNumbers.getLhsContractingDimensions()[0];
  int64_t rhsContractingDimension =
      dotDimensionNumbers.getRhsContractingDimensions()[0];
  if (layout == "rrr") {
    return lhsContractingDimension == 1 && rhsContractingDimension == 0;
  } else if (layout == "rcr") {
    return lhsContractingDimension == 1 && rhsContractingDimension == 1;
  } else if (layout == "crr") {
    return lhsContractingDimension == 0 && rhsContractingDimension == 0;
  } else if (layout == "ccr") {
    return lhsContractingDimension == 0 && rhsContractingDimension == 1;
  }
  return false;
}

bool isNormalizedBmm(mhlo::DotGeneralOp op, const std::string &layout) {
  auto dotDimensionNumbers = op.getDotDimensionNumbers();
  if (dotDimensionNumbers.getLhsBatchingDimensions().size() != 1) {
    return false;
  }
  if (dotDimensionNumbers.getRhsBatchingDimensions().size() != 1) {
    return false;
  }
  if (dotDimensionNumbers.getLhsContractingDimensions().size() != 1) {
    return false;
  }
  if (dotDimensionNumbers.getRhsContractingDimensions().size() != 1) {
    return false;
  }
  auto lhsRank = cast<ShapedType>(op.getLhs().getType()).getRank();
  auto rhsRank = cast<ShapedType>(op.getRhs().getType()).getRank();
  if (lhsRank != 3 || rhsRank != 3) {
    return false;
  }

  int64_t lhsBatchingDimension =
      dotDimensionNumbers.getLhsBatchingDimensions()[0];
  int64_t rhsBatchingDimension =
      dotDimensionNumbers.getRhsBatchingDimensions()[0];
  if (lhsBatchingDimension != 0 || rhsBatchingDimension != 0) {
    return false;
  }
  int64_t lhsContractingDimension =
      dotDimensionNumbers.getLhsContractingDimensions()[0];
  int64_t rhsContractingDimension =
      dotDimensionNumbers.getRhsContractingDimensions()[0];
  if (layout == "rrr") {
    return lhsContractingDimension == 2 && rhsContractingDimension == 1;
  } else if (layout == "rcr") {
    return lhsContractingDimension == 2 && rhsContractingDimension == 2;
  } else if (layout == "crr") {
    return lhsContractingDimension == 1 && rhsContractingDimension == 1;
  } else if (layout == "ccr") {
    return lhsContractingDimension == 1 && rhsContractingDimension == 2;
  }
  return false;
}

LogicalResult isGemmBias(mhlo::AddOp op, const std::string &layout, Value &lhs,
                         Value &rhs, Value &bias) {
  mhlo::DotGeneralOp dotOp = op.getLhs().getDefiningOp<mhlo::DotGeneralOp>();
  mhlo::BroadcastInDimOp bcastOp =
      op.getRhs().getDefiningOp<mhlo::BroadcastInDimOp>();
  if (!dotOp || !bcastOp) {
    dotOp = op.getRhs().getDefiningOp<mhlo::DotGeneralOp>();
    bcastOp = op.getLhs().getDefiningOp<mhlo::BroadcastInDimOp>();
    if (!dotOp || !bcastOp) {
      return failure();
    }
  }
  if (!isNormalizedGemm(dotOp, layout)) {
    return failure();
  }
  if (!dotOp.getResult().hasOneUse()) {
    return failure();
  }
  auto bcastDims =
      llvm::to_vector(bcastOp.getBroadcastDimensions().getValues<int64_t>());
  if (bcastDims.size() != 1 || bcastDims[0] != 1) {
    return failure();
  }

  lhs = dotOp.getLhs();
  rhs = dotOp.getRhs();
  bias = bcastOp.getOperand();
  return success();
}
} // namespace

namespace {

struct ConvertToCatGemm : public OpRewritePattern<mhlo::DotGeneralOp> {
  ConvertToCatGemm(MLIRContext *context, llvm::StringRef layout,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<mhlo::DotGeneralOp>(context, benefit),
        layout(layout.str()) {}
  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    if (!isNormalizedGemm(op, layout)) {
      return failure();
    }
    if (layout == "rrr") {
      rewriter.replaceOpWithNewOp<cat::GemmRRROp>(op, op.getType(), op.getLhs(),
                                                  op.getRhs());
      return success();
    } else if (layout == "rcr") {
      rewriter.replaceOpWithNewOp<cat::GemmRCROp>(op, op.getType(), op.getLhs(),
                                                  op.getRhs());
      return success();
    }
    return failure();
  }

  std::string layout;
};

struct ConvertToCatGemmBias : public OpRewritePattern<mhlo::AddOp> {
  ConvertToCatGemmBias(MLIRContext *context, llvm::StringRef layout,
                       PatternBenefit benefit = 3)
      : OpRewritePattern<mhlo::AddOp>(context, benefit), layout(layout.str()) {}
  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs, rhs, bias;
    if (failed(isGemmBias(op, layout, lhs, rhs, bias))) {
      return failure();
    }

    if (layout == "rrr") {
      rewriter.replaceOpWithNewOp<cat::GemmRRRBiasOp>(op, op.getType(), lhs,
                                                      rhs, bias);
      return success();
    } else if (layout == "rcr") {
      rewriter.replaceOpWithNewOp<cat::GemmRCRBiasOp>(op, op.getType(), lhs,
                                                      rhs, bias);
      return success();
    }
    return failure();
  }

  std::string layout;
};

struct ConvertToCatGemmBiasRelu : public OpRewritePattern<mhlo::MaxOp> {
  ConvertToCatGemmBiasRelu(MLIRContext *context, llvm::StringRef layout,
                           PatternBenefit benefit = 4)
      : OpRewritePattern<mhlo::MaxOp>(context, benefit), layout(layout.str()) {}
  LogicalResult matchAndRewrite(mhlo::MaxOp op,
                                PatternRewriter &rewriter) const override {
    mhlo::AddOp addOp = op.getLhs().getDefiningOp<mhlo::AddOp>();
    mhlo::ConstantOp zero = op.getRhs().getDefiningOp<mhlo::ConstantOp>();
    if (!addOp || !zero) {
      addOp = op.getRhs().getDefiningOp<mhlo::AddOp>();
      zero = op.getLhs().getDefiningOp<mhlo::ConstantOp>();
      if (!addOp || !zero) {
        return failure();
      }
    }
    if (!isZeroAttribute(zero.getValue())) {
      return failure();
    }
    if (!addOp.getResult().hasOneUse()) {
      return failure();
    }

    Value lhs, rhs, bias;
    if (failed(isGemmBias(addOp, layout, lhs, rhs, bias))) {
      return failure();
    }

    if (layout == "rcr") {
      rewriter.replaceOpWithNewOp<cat::GemmRCRBiasReluOp>(op, op.getType(), lhs,
                                                          rhs, bias);
      return success();
    }
    return failure();
  }

  std::string layout;
};

struct ConvertToCatBmm : public OpRewritePattern<mhlo::DotGeneralOp> {
  ConvertToCatBmm(MLIRContext *context, llvm::StringRef layout,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<mhlo::DotGeneralOp>(context, benefit),
        layout(layout.str()) {}
  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    if (!isNormalizedBmm(op, layout)) {
      return failure();
    }
    if (layout == "rrr") {
      rewriter.replaceOpWithNewOp<cat::BMMRRROp>(op, op.getType(), op.getLhs(),
                                                 op.getRhs());
      return success();
    } else if (layout == "rcr") {
      rewriter.replaceOpWithNewOp<cat::BMMRCROp>(op, op.getType(), op.getLhs(),
                                                 op.getRhs());
      return success();
    } else if (layout == "crr") {
      rewriter.replaceOpWithNewOp<cat::BMMCRROp>(op, op.getType(), op.getLhs(),
                                                 op.getRhs());
      return success();
    } else if (layout == "ccr") {
      rewriter.replaceOpWithNewOp<cat::BMMCCROp>(op, op.getType(), op.getLhs(),
                                                 op.getRhs());
      return success();
    }
    return failure();
  }

  std::string layout;
};

struct ConvertToCatBmmAdd : public OpRewritePattern<mhlo::AddOp> {
  ConvertToCatBmmAdd(MLIRContext *context, llvm::StringRef layout,
                     PatternBenefit benefit = 2)
      : OpRewritePattern<mhlo::AddOp>(context, benefit), layout(layout.str()) {}
  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    Value add;
    mhlo::DotGeneralOp dotOp = op.getLhs().getDefiningOp<mhlo::DotGeneralOp>();
    if (!dotOp) {
      dotOp = op.getRhs().getDefiningOp<mhlo::DotGeneralOp>();
      if (!dotOp) {
        return failure();
      }
      add = op.getLhs();
    } else {
      add = op.getRhs();
    }
    if (!dotOp.getResult().hasOneUse()) {
      return failure();
    }

    if (!isNormalizedBmm(dotOp, layout)) {
      return failure();
    }
    if (layout == "rrr") {
      rewriter.replaceOpWithNewOp<cat::BMMRRRAddOp>(
          op, op.getType(), dotOp.getLhs(), dotOp.getRhs(), add);
      return success();
    } else if (layout == "rcr") {
      rewriter.replaceOpWithNewOp<cat::BMMRCRAddOp>(
          op, op.getType(), dotOp.getLhs(), dotOp.getRhs(), add);
      return success();
    } else if (layout == "crr") {
      rewriter.replaceOpWithNewOp<cat::BMMCRRAddOp>(
          op, op.getType(), dotOp.getLhs(), dotOp.getRhs(), add);
      return success();
    } else if (layout == "ccr") {
      rewriter.replaceOpWithNewOp<cat::BMMCCRAddOp>(
          op, op.getType(), dotOp.getLhs(), dotOp.getRhs(), add);
      return success();
    }
    return failure();
  }

  std::string layout;
};

struct ConvertSoftmax : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != getSoftmaxName())
      return failure();
    DictionaryAttr byteirAttrs =
        cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
    if (!byteirAttrs)
      return failure();
    auto axisAttr = cast<IntegerAttr>(byteirAttrs.get("axis"));
    auto newOp = rewriter.create<cat::SoftmaxOp>(
        op.getLoc(), op.getResultTypes()[0], op.getOperands()[0], axisAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertLayerNorm : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != getLayerNormName())
      return failure();
    if (op.getResults().size() != 1)
      return failure();
    DictionaryAttr byteirAttrs =
        cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
    if (!byteirAttrs)
      return failure();

    auto axisAttr = byteirAttrs.getAs<ArrayAttr>("axis");
    assert(axisAttr && "LayerNorm custom call axis attribute not found.");
    auto epsAttr = byteirAttrs.getAs<FloatAttr>("epsilon");
    assert(epsAttr && "LayerNorm custom call epsilon attribute not found.");

    rewriter.replaceOpWithNewOp<cat::LayerNormOp>(
        op, op.getResultTypes()[0], op.getOperands()[0], op.getOperands()[1],
        op.getOperands()[2], axisAttr, epsAttr);
    return success();
  }
};

struct ConvertHloToCatPass : public ConvertHloToCatBase<ConvertHloToCatPass> {
  ConvertHloToCatPass(ArrayRef<std::string> validCatOps) {
    this->validCatOps = validCatOps;
  }

  void runOnOperation() override {
    validCatOpsSet.clear();
    validCatOpsSet.insert(validCatOps.begin(), validCatOps.end());

    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    if (validCatOpsSet.contains("cat.gemm_rrr")) {
      patterns.add<ConvertToCatGemm>(context, "rrr");
    }
    if (validCatOpsSet.contains("cat.gemm_rcr")) {
      patterns.add<ConvertToCatGemm>(context, "rcr");
    }
    if (validCatOpsSet.contains("cat.gemm_rrr_bias")) {
      patterns.add<ConvertToCatGemmBias>(context, "rrr");
    }
    if (validCatOpsSet.contains("cat.gemm_rcr_bias")) {
      patterns.add<ConvertToCatGemmBias>(context, "rcr");
    }
    if (validCatOpsSet.contains("cat.gemm_rcr_bias_relu")) {
      patterns.add<ConvertToCatGemmBiasRelu>(context, "rcr");
    }
    if (validCatOpsSet.contains("cat.bmm_rrr")) {
      patterns.add<ConvertToCatBmm>(context, "rrr");
    }
    if (validCatOpsSet.contains("cat.bmm_rcr")) {
      patterns.add<ConvertToCatBmm>(context, "rcr");
    }
    if (validCatOpsSet.contains("cat.bmm_crr")) {
      patterns.add<ConvertToCatBmm>(context, "crr");
    }
    if (validCatOpsSet.contains("cat.bmm_ccr")) {
      patterns.add<ConvertToCatBmm>(context, "ccr");
    }
    if (validCatOpsSet.contains("cat.bmm_rrr_add")) {
      patterns.add<ConvertToCatBmmAdd>(context, "rrr");
    }
    if (validCatOpsSet.contains("cat.bmm_rcr_add")) {
      patterns.add<ConvertToCatBmmAdd>(context, "rcr");
    }
    if (validCatOpsSet.contains("cat.bmm_crr_add")) {
      patterns.add<ConvertToCatBmmAdd>(context, "crr");
    }
    if (validCatOpsSet.contains("cat.bmm_ccr_add")) {
      patterns.add<ConvertToCatBmmAdd>(context, "ccr");
    }

    if (validCatOpsSet.contains("cat.softmax")) {
      patterns.add<ConvertSoftmax>(context);
    }
    if (validCatOpsSet.contains("cat.layernorm")) {
      patterns.add<ConvertLayerNorm>(context);
    }

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }

  llvm::StringSet<> validCatOpsSet;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertHloToCatPass(ArrayRef<std::string> validCatOps) {
  return std::make_unique<ConvertHloToCatPass>(validCatOps);
}