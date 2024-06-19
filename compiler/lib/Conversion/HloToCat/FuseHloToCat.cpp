//===- FuseHloToCat.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/HloToCat/FuseHloToCat.h"
#include "byteir/Dialect/Cat/IR/CatDialect.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"
#include "./Utils.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace mlir {

namespace {

mlir::StringAttr
GetLayoutFrom3DDotGeneralDimNums(mlir::mhlo::DotDimensionNumbersAttr dims,
                                 Builder *builder) {
  auto ldims = dims.getLhsContractingDimensions();
  auto rdims = dims.getRhsContractingDimensions();
  assert(ldims.size() == 1 && rdims.size() == 1);
  if (ldims[0] == 2 && rdims[0] == 1)
    return builder->getStringAttr("rrr");
  if (ldims[0] == 2 && rdims[0] == 2)
    return builder->getStringAttr("rcr");
  if (ldims[0] == 1 && rdims[0] == 1)
    return builder->getStringAttr("crr");
  if (ldims[0] == 1 && rdims[0] == 2)
    return builder->getStringAttr("ccr");
  llvm_unreachable("unsupported dot dimension_numbers");
}

mlir::StringAttr
GetLayoutFromConvDimNums(mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers,
                         Builder *builder) {
  auto layoutStr = getConvLayoutString(dimension_numbers);
  return builder->getStringAttr(layoutStr);
}

#include "FuseHloToCatPattern.inc"

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
    auto resultTy = cast<ShapedType>(op.getResultTypes()[0]);
    auto inputTy = cast<ShapedType>(op.getOperands()[0].getType());
    auto softmaxTy = resultTy;
    bool needConvert = resultTy.getElementType() != inputTy.getElementType();
    if (needConvert) {
      softmaxTy =
          RankedTensorType::get(resultTy.getShape(), inputTy.getElementType());
    }
    auto newOp = rewriter.create<cat::SoftmaxOp>(op.getLoc(), softmaxTy,
                                                 op.getOperands()[0], axisAttr);
    if (needConvert) {
      auto castOp = rewriter.create<mhlo::ConvertOp>(op.getLoc(), resultTy,
                                                     newOp.getResult());
      rewriter.replaceOp(op, castOp.getResult());
    } else {
      rewriter.replaceOp(op, newOp.getResult());
    }
    return success();
  }
};

struct ConvertLayerNorm : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != getLayerNormName())
      return failure();
    DictionaryAttr byteirAttrs =
        cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
    if (!byteirAttrs)
      return failure();
    if (op.getResults().size() > 1)
      return failure();
    auto axisAttr = byteirAttrs.getAs<ArrayAttr>("axis");
    assert(axisAttr && "LayerNorm custom call axis attribute not found.");

    auto epsAttr = byteirAttrs.getAs<FloatAttr>("epsilon");
    assert(epsAttr && "LayerNorm custom call epsilon attribute not found.");

    auto newOp = rewriter.create<cat::LayerNormOp>(
        op.getLoc(), op.getResultTypes()[0], op.getOperands()[0],
        op.getOperands()[1], op.getOperands()[2], axisAttr, epsAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// dot(transpose(x), y) => bmm_crr(x, y)
// there is no gemm_crr op in ait, so we use bmm_crr
struct ConvertTransposeGemmRrrToBmmCrr
    : public OpRewritePattern<cat::GemmRRROp> {
  using OpRewritePattern<cat::GemmRRROp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cat::GemmRRROp op,
                                PatternRewriter &rewriter) const override {
    auto transpose_op = op.getLhs().getDefiningOp<mhlo::TransposeOp>();
    if (!transpose_op) {
      return failure();
    }
    SmallVector<int64_t> permutation;
    getValuesFromDenseIntElementsAttr(transpose_op.getPermutation(),
                                      permutation);
    if (permutation.size() != 2) {
      return failure();
    }
    if (permutation[0] != 1 || permutation[1] != 0) {
      return failure();
    }
    auto lhs = transpose_op.getOperand();
    auto rhs = op.getRhs();
    int64_t batch_shape = 1;
    // build reshape lhs op
    auto lhs_shape = lhs.getType().getShape();
    RankedTensorType lhs_reshaped_type =
        RankedTensorType::get({batch_shape, lhs_shape[0], lhs_shape[1]},
                              lhs.getType().getElementType());
    auto lhsReshapeOp =
        rewriter.create<mhlo::ReshapeOp>(op.getLoc(), lhs_reshaped_type, lhs);
    // build reshape rhs op
    auto rhs_shape = rhs.getType().getShape();
    RankedTensorType rhs_reshaped_type =
        RankedTensorType::get({batch_shape, rhs_shape[0], rhs_shape[1]},
                              rhs.getType().getElementType());
    auto rhsReshapeOp =
        rewriter.create<mhlo::ReshapeOp>(op.getLoc(), rhs_reshaped_type, rhs);
    // build cat bmm crr op
    RankedTensorType bmm_output_type =
        RankedTensorType::get({batch_shape, lhs_shape[1], rhs_shape[1]},
                              rhs.getType().getElementType());
    auto bmmCrrOp = rewriter.create<cat::BMMCRROp>(op.getLoc(), bmm_output_type,
                                                   lhsReshapeOp, rhsReshapeOp);
    auto newOp = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), op.getResult().getType(), bmmCrrOp);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// bmm_rrr(x, reshape(transpose(y, [0,1,3,2]))) -> bmm_rcr(x, reshape(y))
struct ConvertTransposeReshapeBmmRrrToBmmRcr
    : public OpRewritePattern<cat::BMMRRROp> {
  using OpRewritePattern<cat::BMMRRROp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cat::BMMRRROp op,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp = op.getRhs().getDefiningOp<mhlo::ReshapeOp>();
    if (!reshapeOp) {
      return failure();
    }
    auto transposeOp =
        reshapeOp.getOperand().getDefiningOp<mhlo::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }
    SmallVector<int64_t> permutation;
    getValuesFromDenseIntElementsAttr(transposeOp.getPermutation(),
                                      permutation);
    auto transposeIn = transposeOp.getOperand(); // transpose_in = rhs
    auto transposeInShape = transposeIn.getType().getShape();
    auto lhs = op.getLhs();
    if (permutation.size() != 4) {
      return failure();
    }
    if (permutation[0] != 0 || permutation[1] != 1 || permutation[2] != 3 ||
        permutation[3] != 2) {
      return failure();
    }
    auto reshapeOperandShape =
        cast<ShapedType>(reshapeOp.getOperand().getType()).getShape();
    auto reshapeResultShape =
        cast<ShapedType>(reshapeOp.getResult().getType()).getShape();
    if (reshapeResultShape.size() != 3) {
      return failure();
    }
    if (reshapeResultShape[0] !=
            reshapeOperandShape[0] * reshapeOperandShape[1] ||
        reshapeResultShape[1] != reshapeOperandShape[2] ||
        reshapeResultShape[2] != reshapeOperandShape[3]) {
      return failure();
    }
    // build reshape rhs op
    RankedTensorType transposeInReshapedType =
        RankedTensorType::get({transposeInShape[0] * transposeInShape[1],
                               transposeInShape[2], transposeInShape[3]},
                              transposeIn.getType().getElementType());
    auto rhsReshapeOp = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), transposeInReshapedType, transposeIn);
    // build cat bmm rcr op
    auto newOp = rewriter.create<cat::BMMRCROp>(op.getLoc(), op.getType(), lhs,
                                                rhsReshapeOp);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

template <typename SrcBmmType, typename DstBmmType>
struct ConvertBmmReshapeTransposeToBmmReshape
    : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp = op.getOperand().getDefiningOp<mhlo::ReshapeOp>();
    if (!reshapeOp || !reshapeOp.getResult().hasOneUse()) {
      return failure();
    }
    auto srcBmmOp = reshapeOp.getOperand().getDefiningOp<SrcBmmType>();
    if (!srcBmmOp || !srcBmmOp.getResult().hasOneUse()) {
      return failure();
    }
    SmallVector<int64_t> permutation;
    getValuesFromDenseIntElementsAttr(op.getPermutation(), permutation);
    if (permutation.size() != 4) {
      return failure();
    }
    if (permutation[0] != 0 || permutation[1] != 1 || permutation[2] != 3 ||
        permutation[3] != 2) {
      return failure();
    }
    auto reshapeOperandShape =
        cast<ShapedType>(reshapeOp.getOperand().getType()).getShape();
    auto reshapeResultShape =
        cast<ShapedType>(reshapeOp.getResult().getType()).getShape();
    if (reshapeOperandShape.size() != 3) {
      return failure();
    }
    if (reshapeOperandShape[0] !=
            reshapeResultShape[0] * reshapeResultShape[1] ||
        reshapeOperandShape[1] != reshapeResultShape[2] ||
        reshapeOperandShape[2] != reshapeResultShape[3]) {
      return failure();
    }

    auto srcBmmOpType = cast<ShapedType>(srcBmmOp.getType());
    // build dst bmm op
    RankedTensorType dstBmmOpResultType = RankedTensorType::get(
        {srcBmmOpType.getDimSize(0), srcBmmOpType.getDimSize(2),
         srcBmmOpType.getDimSize(1)},
        srcBmmOpType.getElementType());
    auto dstBmmOp = rewriter.create<DstBmmType>(
        op.getLoc(), dstBmmOpResultType, srcBmmOp.getLhs(), srcBmmOp.getRhs());
    // build new reshape op
    auto newShapeOp = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), op.getType(), dstBmmOp.getResult());
    rewriter.replaceOp(op, newShapeOp.getResult());
    return success();
  }
};

// bmm_rrr(x, broadcast_in_dim(y)) => reshape(gemm_rrr(reshape(x), y))
struct ConvertBmmRRRBroadcastToReshapeGemmRRRReshape
    : public OpRewritePattern<cat::BMMRRROp> {
  using OpRewritePattern<cat::BMMRRROp>::OpRewritePattern;
  LogicalResult matchAndRewrite(cat::BMMRRROp op,
                                PatternRewriter &rewriter) const override {
    auto bCastOp = op.getRhs().getDefiningOp<mhlo::BroadcastInDimOp>();
    if (!bCastOp) {
      return failure();
    }
    auto lhsType = cast<ShapedType>(op.getLhs().getType());
    auto rhsType = cast<ShapedType>(op.getRhs().getType());
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
      return failure();
    }
    SmallVector<int64_t> broadcastDimensions;
    getValuesFromDenseIntElementsAttr(bCastOp.getBroadcastDimensions(),
                                      broadcastDimensions);
    if (broadcastDimensions.size() != 2) {
      return failure();
    }
    if (broadcastDimensions[0] != 1 || broadcastDimensions[1] != 2) {
      return failure();
    }

    RankedTensorType firstReshapeType = RankedTensorType::get(
        {lhsType.getDimSize(0) * lhsType.getDimSize(1), lhsType.getDimSize(2)},
        lhsType.getElementType());
    RankedTensorType gemmType = RankedTensorType::get(
        {firstReshapeType.getDimSize(0), rhsType.getDimSize(2)},
        lhsType.getElementType());
    auto firstReshape = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), firstReshapeType, op.getLhs());
    auto gemm = rewriter.create<cat::GemmRRROp>(
        op.getLoc(), gemmType, firstReshape, bCastOp.getOperand());
    auto secondReshape =
        rewriter.create<mhlo::ReshapeOp>(op.getLoc(), op.getType(), gemm);
    rewriter.replaceOp(op, secondReshape);
    return success();
  }
};

struct FuseMhloToCatPass : public FuseMhloToCatBase<FuseMhloToCatPass> {
public:
  FuseMhloToCatPass() = default;
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    // ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateFuseMhloToCatPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateFuseMhloToCatPattern(RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
  // clang-format off
  patterns.add<ConvertSoftmax,
               ConvertLayerNorm,
               ConvertTransposeGemmRrrToBmmCrr,
               ConvertTransposeReshapeBmmRrrToBmmRcr,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMRRROp, cat::BMMRRCOp>,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMRCROp, cat::BMMRCCOp>,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMCRROp, cat::BMMCRCOp>,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMCCROp, cat::BMMCCCOp>,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMRRCOp, cat::BMMRRROp>,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMRCCOp, cat::BMMRCROp>,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMCRCOp, cat::BMMCRROp>,
               ConvertBmmReshapeTransposeToBmmReshape<cat::BMMCCCOp, cat::BMMCCROp>,
               ConvertBmmRRRBroadcastToReshapeGemmRRRReshape
               >(patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> createFuseMhloToCatPass() {
  return std::make_unique<FuseMhloToCatPass>();
}

} // namespace mlir
