//===- HloFolder.cpp ------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"

#include "byteir/Dialect/mhlo/Analysis/DimFromBroadcast.h"
#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace mlir;
using namespace llvm;
using namespace ::byteir;
using namespace mlir::mhlo;

#define K_INITIAL -999

namespace {

//===----------------------------------------------------------------------===//
// Add + Scatter => Scatter Pattern
//===----------------------------------------------------------------------===//

static LogicalResult
AddScatterAddMatchAndRewriteHelper(mhlo::AddOp addOp, int idx,
                                   PatternRewriter &rewriter) {

  // Match
  mhlo::ScatterOp scatterOp =
      addOp.getOperand(idx).getDefiningOp<mhlo::ScatterOp>();

  if (!scatterOp) {
    return failure();
  }

  // check wthether scatter supported
  Region &region = scatterOp.getUpdateComputation();
  // only support single block
  if (region.getBlocks().size() != 1) {
    return failure();
  }
  // only support one operand one update
  if (scatterOp.getInputs().size() != 1 || scatterOp.getUpdates().size() != 1 ||
      scatterOp->getNumResults() != 1) {
    return failure();
  }

  auto &block = region.front();
  if (!isBlockSingleOp<mhlo::AddOp>(&block)) {
    return failure();
  }

  Value initialVal = scatterOp.getInputs()[0];
  if (!isSplatMhloConstantValue(initialVal, (int64_t)0) &&
      !isSplatMhloConstantValue(initialVal, 0.0)) {
    return failure();
  }

  // Rewrite
  int anotherIdx = 1 - idx;
  auto cloned = rewriter.clone(*scatterOp.getOperation());
  cloned->setOperand(0, addOp.getOperand(anotherIdx));
  rewriter.replaceOp(addOp, cloned->getResult(0));
  return success();
}

// Add + Scatter {add} -> Scatter
// TODO other scatter support
struct AddScatterAddToScatterPattern : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter &rewriter) const override {

    // handle left
    if (failed(AddScatterAddMatchAndRewriteHelper(op, 0, rewriter))) {
      // handle right
      return AddScatterAddMatchAndRewriteHelper(op, 1, rewriter);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// RemoveTrivialTorchIndexSelect Pattern
//===----------------------------------------------------------------------===//

struct RemoveTrivialTorchIndexSelect
    : public OpRewritePattern<mhlo::TorchIndexSelectOp> {
  RemoveTrivialTorchIndexSelect(MLIRContext *context, DimFlagAnalysis *analys)
      : OpRewritePattern<mhlo::TorchIndexSelectOp>(context), analysis(analys) {}

  LogicalResult matchAndRewrite(mhlo::TorchIndexSelectOp op,
                                PatternRewriter &rewriter) const override {
    uint64_t dim = op.getDim();
    uint64_t batchDims = op.getBatchDims();
    Value index = op.getIndex();
    Value input = op.getOperand();

    auto indexShapedType = index.getType().dyn_cast<ShapedType>();
    auto inputShapedType = input.getType().dyn_cast<ShapedType>();
    if (batchDims > 0 || indexShapedType.getRank() != 1 || !indexShapedType ||
        !indexShapedType.hasStaticShape() || !inputShapedType ||
        !inputShapedType.hasStaticShape() ||
        indexShapedType.getShape()[0] != inputShapedType.getShape()[dim]) {
      return failure();
    }

    SmallVector<bool> fromBroadcast = analysis->getDimFlag(input);
    if (!(int64_t(fromBroadcast.size()) == inputShapedType.getRank()) ||
        !fromBroadcast[dim]) {
      return failure();
    }
    rewriter.replaceOp(op, input);
    return success();
  }

  DimFlagAnalysis *analysis;
};

//===----------------------------------------------------------------------===//
// PadConvolution Pattern
//===----------------------------------------------------------------------===//

struct PadConvToConvPattern : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    auto padOp = op.getLhs().getDefiningOp<mhlo::PadOp>();
    if (!padOp || !isZeroAttribute(padOp.getInteriorPadding())) {
      return failure();
    }
    auto constOp = padOp.getPaddingValue().getDefiningOp<mhlo::ConstantOp>();
    if (!constOp || !isZeroAttribute(constOp.getValue())) {
      return failure();
    }

    const auto edgePaddingLow = padOp.getEdgePaddingLow().getValues<int64_t>();
    const auto edgePaddingHigh =
        padOp.getEdgePaddingHigh().getValues<int64_t>();
    auto dimensionNumbers = op.getDimensionNumbers();
    auto inputSpatialDims = dimensionNumbers.getInputSpatialDimensions();
    llvm::SmallDenseSet<int64_t> inputSpatialDimsSet(inputSpatialDims.begin(),
                                                     inputSpatialDims.end());
    for (size_t i = 0; i < edgePaddingLow.size(); i++) {
      if (!inputSpatialDimsSet.contains(i)) {
        if (edgePaddingLow[i] != 0 || edgePaddingHigh[i] != 0) {
          return failure();
        }
      }
    }

    SmallVector<int64_t> oldPadding(inputSpatialDims.size() * 2, 0);
    if (op.getPadding().has_value()) {
      oldPadding =
          SmallVector<int64_t>(op.getPaddingAttr().getValues<int64_t>().begin(),
                               op.getPaddingAttr().getValues<int64_t>().end());
    }
    SmallVector<int64_t> newPadding;
    for (size_t i = 0; i < inputSpatialDims.size(); i++) {
      newPadding.push_back(edgePaddingLow[inputSpatialDims[i]] +
                           oldPadding[i * 2]);
      newPadding.push_back(edgePaddingHigh[inputSpatialDims[i]] +
                           oldPadding[i * 2 + 1]);
    }
    auto newPaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(inputSpatialDims.size()), 2},
            rewriter.getI64Type()),
        newPadding);

    auto newOp = cast<mhlo::ConvolutionOp>(rewriter.clone(*op));
    newOp.setOperand(0, padOp.getOperand());
    newOp.setPaddingAttr(newPaddingAttr);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvFollowedByMulOrAdd Pattern
// TODO: handle similar cases of dot op followed by mul or add
//===----------------------------------------------------------------------===//

// Return the expanded constOp if applicable, return std::nullopt if not.
// Applicable if all following constraint satisfied:
// 1. the op's input has static shape
// 2. op's input rank equals 1, or it is equal to output rank
// 3. there's at most one dim in input shape whose size is not equal to 1, and
//     it should be euqal to featureDim
// 4. the input's DefiningOp is of type mhlo::ConstantOp
// 5. the const op's attr is of type DenseElementsAttr
std::optional<ConstantOp> getBroadcastedConstOp(BroadcastInDimOp op,
                                                int64_t featureDim) {
  Value broadInDimInput = op.getOperand();
  ShapedType broadInDimInpShape = broadInDimInput.getType().cast<ShapedType>();
  Value broadInDimOutput = op->getResult(0);
  ShapedType broadInDimOupShape = broadInDimOutput.getType().cast<ShapedType>();

  // Only need to check the input shape of broadcast_in_dim
  if (!broadInDimInpShape.hasStaticShape())
    return std::nullopt;

  // op's input rank equals 1, or it is equal to output rank
  if (broadInDimInpShape.getRank() == 1) {
    SmallVector<int64_t> broadcastDims;
    int64_t bdim = (*op.getBroadcastDimensions().begin()).getSExtValue();
    if (featureDim != bdim)
      return std::nullopt;
  } else if (broadInDimInpShape.getRank() == broadInDimOupShape.getRank()) {
    int64_t nonOneDim = K_INITIAL;
    for (int64_t i = 0; i < broadInDimInpShape.getRank(); ++i) {
      int64_t dimSize = broadInDimInpShape.getDimSize(i);
      if (dimSize != 1) {
        if (nonOneDim >= 0)
          return std::nullopt;
        else {
          nonOneDim = i;
        }
      }
    }

    // There's at most one dim whose size is not equal to 1, and it should be
    // euqal to featureDim.
    if (nonOneDim != K_INITIAL && nonOneDim != featureDim)
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  auto constOp = dyn_cast_or_null<ConstantOp>(broadInDimInput.getDefiningOp());
  if (!constOp)
    return std::nullopt;

  if (!constOp.getValue().dyn_cast_or_null<DenseElementsAttr>())
    return std::nullopt;

  return constOp;
}

struct ConvOrConvBiasFollowedByBroadcastOp
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  using OpRewritePattern<mhlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp convOp,
                                PatternRewriter &rewriter) const override {
    Value convOrBiasOut = convOp->getResult(0);
    if (!convOrBiasOut.hasOneUse())
      return failure();

    Operation *convOrBiasUser = *convOrBiasOut.user_begin();
    int64_t featureDim =
        convOp.getDimensionNumbers().getOutputFeatureDimension();
    Value convWeight = convOp.getRhs();

    if (!convWeight.getDefiningOp() ||
        !isa<ConstantOp>(convWeight.getDefiningOp()))
      return failure();
    if (!convWeight.hasOneUse())
      return failure();
    if (!convWeight.getType().cast<ShapedType>().hasStaticShape())
      return failure();

    // handle the conv + bias scenario
    auto biasAddOp = dyn_cast_or_null<mhlo::AddOp>(convOrBiasUser);
    ConstantOp biasConst = nullptr;
    BroadcastInDimOp biasBroadcastInDimOp = nullptr;
    if (biasAddOp) {
      // Here we update `convOrBiasOut` and `convOrBiasUser`
      convOrBiasOut = biasAddOp->getResult(0);
      if (!convOrBiasOut.hasOneUse())
        return failure();
      convOrBiasUser = *convOrBiasOut.user_begin();

      unsigned convOperandNumber =
          convOp->getResult(0).use_begin()->getOperandNumber();
      assert(convOperandNumber < 2);
      auto broadInDimOp = biasAddOp->getOperand(1 - convOperandNumber)
                              .getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.has_value())
        return failure();

      biasConst = *maybeConstOp;
      biasBroadcastInDimOp = broadInDimOp;
    }

    unsigned convOrBiasOperandNumber =
        convOrBiasOut.use_begin()->getOperandNumber();

    if (auto scaleOp = dyn_cast_or_null<MulOp>(convOrBiasUser)) {
      auto broadInDimOp = scaleOp->getOperand(1 - convOrBiasOperandNumber)
                              .getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.has_value())
        return failure();
      ConstantOp constOp = *maybeConstOp;

      // Start to construct a new subgraph which could be const folded.
      if (!constOp->isBeforeInBlock(convOp))
        constOp->moveBefore(convOp);

      // construct new conv weight
      OpBuilder builder(convOp);
      auto convWeightType = convOp.getRhs().getType().cast<ShapedType>();
      auto weightFeatureDim =
          convOp.getDimensionNumbers().getKernelOutputFeatureDimension();
      ReshapeOp newReshapeOp = builder.create<mhlo::ReshapeOp>(
          constOp->getLoc(),
          RankedTensorType::get({convWeightType.getDimSize(weightFeatureDim)},
                                convWeightType.getElementType()),
          constOp.getOutput());
      BroadcastInDimOp newBroadInDimOp = builder.create<mhlo::BroadcastInDimOp>(
          constOp->getLoc(), convWeightType, newReshapeOp->getResult(0),
          rewriter.getI64TensorAttr({weightFeatureDim}));
      MulOp newMulOp = builder.create<MulOp>(constOp->getLoc(), convWeight,
                                             newBroadInDimOp->getResult(0));
      convOp->setOperand(1, newMulOp->getResult(0));

      // construct new conv bias
      if (biasAddOp) {
        OpBuilder builder(biasAddOp);
        ReshapeOp newReshapeOp = builder.create<mhlo::ReshapeOp>(
            constOp->getLoc(), biasConst.getOutput().getType(),
            constOp.getOutput());
        MulOp newMulOp =
            builder.create<MulOp>(constOp->getLoc(), biasConst.getOutput(),
                                  newReshapeOp->getResult(0));
        biasBroadcastInDimOp->setOperand(0, newMulOp->getResult(0));
      }

      // update conv's uses
      scaleOp->getResult(0).replaceAllUsesWith(convOrBiasOut);

    } else if (auto offsetOp = dyn_cast_or_null<AddOp>(convOrBiasUser)) {
      auto broadInDimOp = offsetOp->getOperand(1 - convOrBiasOperandNumber)
                              .getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.has_value())
        return failure();
      ConstantOp constOp = *maybeConstOp;

      // Start to construct a new subgraph which could be const folded.
      if (!constOp->isBeforeInBlock(convOp))
        constOp->moveBefore(convOp);

      // construct new conv bias
      assert(biasAddOp);
      OpBuilder builder(biasAddOp);
      ReshapeOp newReshapeOp = builder.create<mhlo::ReshapeOp>(
          constOp->getLoc(), biasConst.getOutput().getType(),
          constOp.getOutput());
      AddOp newAddOp = builder.create<AddOp>(
          constOp->getLoc(), biasConst.getOutput(), newReshapeOp->getResult(0));
      biasBroadcastInDimOp->setOperand(0, newAddOp->getResult(0));

      // update conv's uses
      offsetOp->getResult(0).replaceAllUsesWith(convOrBiasOut);

    } else if (auto subOp = dyn_cast_or_null<SubtractOp>(convOrBiasUser)) {
      // conv_or_bias - a => conv_or_bias + (- a)

      // b_const should be rhs
      auto broadInDimOp =
          subOp.getRhs().getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.has_value())
        return failure();
      ConstantOp constOp = *maybeConstOp;

      OpBuilder builder(subOp);
      // replace b_const with (- b_const)
      NegOp negOp = builder.create<mhlo::NegOp>(constOp->getLoc(),
                                                constOp.getOutput().getType(),
                                                constOp.getOutput());
      negOp->moveBefore(broadInDimOp);
      broadInDimOp->setOperand(0, negOp.getResult());

      // replace mhlo.sub with mhlo.add
      AddOp addOp = builder.create<mhlo::AddOp>(subOp->getLoc(),
                                                subOp.getResult().getType(),
                                                subOp.getLhs(), subOp.getRhs());
      subOp.getResult().replaceAllUsesWith(addOp.getResult());

    } else if (auto divOp = dyn_cast_or_null<DivOp>(convOrBiasUser)) {
      // conv_or_bias / a => conv_or_bias * (1 / a)

      // b_const should be rhs
      auto broadInDimOp =
          divOp.getRhs().getDefiningOp<mhlo::BroadcastInDimOp>();
      if (!broadInDimOp)
        return failure();

      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.has_value())
        return failure();
      ConstantOp constOp = *maybeConstOp;

      OpBuilder builder(divOp);
      // replace b_const with 1 / b_const
      auto constType = constOp.getOutput().getType().cast<RankedTensorType>();
      auto fpType = constType.getElementType().dyn_cast<FloatType>();
      if (!fpType) {
        return failure();
      }
      llvm::APFloat one(static_cast<double>(1));
      bool losesInfo; // didn't check this
      one.convert(fpType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
      ConstantOp constOne = builder.create<mhlo::ConstantOp>(
          constOp->getLoc(), DenseFPElementsAttr::get(constType, one));
      constOne->moveBefore(broadInDimOp);
      DivOp oneDiv = builder.create<mhlo::DivOp>(constOp->getLoc(), constType,
                                                 constOne.getOutput(),
                                                 constOp.getOutput());
      oneDiv->moveBefore(broadInDimOp);
      broadInDimOp->setOperand(0, oneDiv.getResult());

      // replace mhlo.div with mhlo.mul
      MulOp mulOp = builder.create<mhlo::MulOp>(divOp->getLoc(),
                                                divOp.getResult().getType(),
                                                divOp.getLhs(), divOp.getRhs());
      divOp.getResult().replaceAllUsesWith(mulOp.getResult());

    } else {
      return failure();
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// PadReduceWindowToReduceWindow Pattern
//===----------------------------------------------------------------------===//

struct PadReduceWindowToReduceWindowPattern
    : public OpRewritePattern<mhlo::ReduceWindowOp> {
  using OpRewritePattern<mhlo::ReduceWindowOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1 ||
        op.getResults().size() != 1) {
      return failure();
    }
    // handle a common, special case of ReduceWindow for 1 input, 1 init_values,
    // and 1 result
    if (auto pad = dyn_cast_or_null<mhlo::PadOp>(
            op.getOperands().front().getDefiningOp())) {
      if (pad.getPaddingValue() == op.getInitValues().front() &&
          isZeroAttribute(pad.getInteriorPadding())) {
        // create a padding
        const auto edge_padding_low =
            pad.getEdgePaddingLow().getValues<int64_t>();
        const auto edge_padding_high =
            pad.getEdgePaddingHigh().getValues<int64_t>();
        SmallVector<int64_t> oldPadding(edge_padding_low.size() * 2, 0);
        if (op.getPadding().has_value()) {
          oldPadding = SmallVector<int64_t>(
              op.getPaddingAttr().getValues<int64_t>().begin(),
              op.getPaddingAttr().getValues<int64_t>().end());
        }
        SmallVector<int64_t> newPadding;
        for (size_t i = 0; i < edge_padding_low.size(); i++) {
          newPadding.push_back(oldPadding[i * 2] + edge_padding_low[i]);
          newPadding.push_back(oldPadding[i * 2 + 1] + edge_padding_high[i]);
        }

        auto newPaddingAttr = DenseIntElementsAttr::get(
            RankedTensorType::get(
                {static_cast<int64_t>(edge_padding_low.size()), 2},
                rewriter.getI64Type()),
            newPadding);

        auto newOp = cast<mhlo::ReduceWindowOp>(rewriter.clone(*op));
        newOp.setOperand(0, pad.getOperand());
        newOp.setPaddingAttr(newPaddingAttr);
        rewriter.replaceOp(op, newOp->getResult(0));
        return success();
      }
    }

    return failure();
  }
};

struct HloFolderPass : public HloFolderBase<HloFolderPass> {
  void runOnOperation() override {
    DimFromBroadcast dim_from_broadcast;
    DimFlagAnalysis dim_from_broadcast_analysis(&dim_from_broadcast);
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateHloFoldPatterns(patterns);
    patterns.add<RemoveTrivialTorchIndexSelect>(context,
                                                &dim_from_broadcast_analysis);
    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, context);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateHloFoldPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<AddScatterAddToScatterPattern, 
               ConvOrConvBiasFollowedByBroadcastOp,
               PadConvToConvPattern, 
               PadReduceWindowToReduceWindowPattern>(
          patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createHloFolderPass() {
  return std::make_unique<HloFolderPass>();
}
