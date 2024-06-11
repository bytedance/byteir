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

#include "PassDetail.h"
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
#include "llvm/Support/Debug.h"
#include <utility>

using namespace mlir;
using namespace llvm;
using namespace ::byteir;
using namespace mlir::mhlo;

#define DEBUG_TYPE "hlo-folder"
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

    auto indexShapedType = dyn_cast<ShapedType>(index.getType());
    auto inputShapedType = dyn_cast<ShapedType>(input.getType());
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
// ConvOrDotWithBiasFollowedByBroadcast Pattern
//===----------------------------------------------------------------------===//

// Represent 3 type of constant operand.
// 1. constant operand is expand by broadcastInDim explicitly.
// 2. The shape of the constant operands is 1 in all dimensions other than
// featureDim, such as (1x1x3x1), where featureDim=2.
// 3. the constant operand is a splat tensor
enum constOperandType {
  vectorWithBroadcast,
  vectorWithoutBroadcast,
  splatTensor
};

// Return the expanded constOp if applicable, return std::nullopt if not.
// Applicable if all following constraint satisfied:
// 1. the broadcastDimensions's size should be 1 and equal to featureDim
// 2. the input's DefiningOp is of type mhlo::ConstantOp
// 3. the const op's attr is of type DenseElementsAttr
std::optional<ConstantOp> getBroadcastedConstOp(BroadcastInDimOp op,
                                                int64_t featureDim) {
  if (!op) {
    return std::nullopt;
  }
  if (op.getBroadcastDimensions().size() != 1 ||
      (*op.getBroadcastDimensions().begin()).getSExtValue() != featureDim) {
    return std::nullopt;
  }
  auto constOp = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
  if (!constOp || !constOp.getValue().isa<DenseElementsAttr>()) {
    return std::nullopt;
  }
  return constOp;
}

// Return the constOp and "constOperandType" if applicable, return std::nullopt
// if not.
// For "vectorWithBroadcast" type, applicable mean 1&2&3:
// For "vectorWithoutBroadcast" type, applicable mean 2&3&4:
// For "vectorWithBroadcast" type, applicable mean 2&5:
// 1. the broadcastDimensions's size should be 1 and equal to featureDim
// 2. the input's DefiningOp is of type mhlo::ConstantOp
// 3. the const op's attr is of type DenseElementsAttr
// 4. the const op's shape is 1 in all dimensions other than featureDim
// 5. the const op is splat
std::optional<std::pair<ConstantOp, constOperandType>>
getConstOpWithType(Value op, int64_t featureDim) {
  auto broadcastInDimOp = op.getDefiningOp<mhlo::BroadcastInDimOp>();
  mhlo::ConstantOp constOp;
  if (broadcastInDimOp) {
    if (broadcastInDimOp.getBroadcastDimensions().size() != 1 ||
        (*broadcastInDimOp.getBroadcastDimensions().begin()).getSExtValue() !=
            featureDim) {
      return std::nullopt;
    }
    constOp = broadcastInDimOp.getOperand().getDefiningOp<mhlo::ConstantOp>();
  } else {
    constOp = op.getDefiningOp<mhlo::ConstantOp>();
  }
  if (!constOp || !constOp.getValue().isa<DenseElementsAttr>()) {
    return std::nullopt;
  }

  constOperandType constType;
  if (broadcastInDimOp) {
    constType = vectorWithBroadcast;
  } else if (constOp.getValue().isSplat()) {
    constType = splatTensor;
  } else {
    auto constOpShapeType = constOp.getValue().getShapedType();
    if (constOpShapeType.getNumElements() !=
        constOpShapeType.getDimSize(featureDim)) {
      return std::nullopt;
    }
    constType = vectorWithoutBroadcast;
  }

  return std::make_pair(constOp, constType);
}

// handle vectorWithoutBroadcast and splatTensor to vector-constant
mhlo::ConstantOp createVectorConstFromConst(int64_t featureDim,
                                            mhlo::ConstantOp constOp,
                                            PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(constOp.getResult().getParentBlock());
  ShapedType attrType = constOp.getValue().getShapedType();
  ShapedType newAttrType =
      attrType.clone(llvm::ArrayRef({attrType.getShape()[featureDim]}));
  if (constOp.getValue().isSplat()) {
    return rewriter.create<mhlo::ConstantOp>(
        constOp.getLoc(),
        cast<DenseElementsAttr>(constOp.getValue()).resizeSplat(newAttrType));
  } else {
    return rewriter.create<mhlo::ConstantOp>(
        constOp.getLoc(),
        cast<DenseElementsAttr>(constOp.getValue()).reshape(newAttrType));
  }
}

template <typename OpTy, typename = void>
struct ConvOrDotWithBiasFollowedByBroadcastPattern;

template <typename OpTy>
struct ConvOrDotWithBiasFollowedByBroadcastPattern<
    OpTy, std::enable_if_t<
              llvm::is_one_of<OpTy, mhlo::ConvolutionOp, mhlo::DotOp>::value>>
    : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy convOrDotOp,
                                PatternRewriter &rewriter) const override {
    Value convOrDotOrBiasOut = convOrDotOp->getResult(0);
    if (!convOrDotOrBiasOut.hasOneUse())
      return failure();

    Operation *convOrDotOrBiasUser = *convOrDotOrBiasOut.user_begin();
    int64_t weightFeatureDim = 1;
    int64_t featureDim = 1;
    static_assert(
        llvm::is_one_of<OpTy, mhlo::ConvolutionOp, mhlo::DotOp>::value &&
        "The operation type is unexpected");
    if (std::is_same_v<OpTy, mhlo::ConvolutionOp>) {
      auto convOp = dyn_cast<mhlo::ConvolutionOp>(&convOrDotOp);
      weightFeatureDim =
          convOp->getDimensionNumbers().getKernelOutputFeatureDimension();
      featureDim = convOp->getDimensionNumbers().getOutputFeatureDimension();
    } else if (std::is_same_v<OpTy, mhlo::DotOp>) {
      auto dotOp = dyn_cast<mhlo::DotOp>(&convOrDotOp);
      int64_t dotLhsRank =
          cast<ShapedType>(dotOp->getLhs().getType()).getRank();
      int64_t dotRhsRank =
          cast<ShapedType>(dotOp->getRhs().getType()).getRank();
      if ((dotLhsRank != 2) || (dotRhsRank != 2)) {
        return failure();
      }
    }

    Value weight = convOrDotOp.getRhs();
    if (!isDenseMhloConstantValue(weight))
      return failure();
    if (!weight.hasOneUse())
      return failure();
    if (!cast<ShapedType>(weight.getType()).hasStaticShape())
      return failure();

    // handle the conv/dot + bias scenario
    auto biasAddOp = dyn_cast_or_null<mhlo::AddOp>(convOrDotOrBiasUser);
    Value biasOperand;
    int64_t biasConstOperandNumber = -1;

    if (biasAddOp) {
      // Here we update `convOrDotOrBiasOut` and `convOrDotOrBiasUser`
      convOrDotOrBiasOut = biasAddOp->getResult(0);
      if (!convOrDotOrBiasOut.hasOneUse())
        return failure();
      convOrDotOrBiasUser = *convOrDotOrBiasOut.user_begin();

      int64_t convOrDotOperandNumber = static_cast<int64_t>(
          convOrDotOp->getResult(0).use_begin()->getOperandNumber());
      assert(convOrDotOperandNumber < 2);
      biasConstOperandNumber = 1 - convOrDotOperandNumber;

      biasOperand = biasAddOp->getOperand(biasConstOperandNumber);
      auto maybeConstOpWithType = getConstOpWithType(biasOperand, featureDim);
      if (!maybeConstOpWithType.has_value()) {
        return failure();
      }
    }

    unsigned convOrDotOrBiasOperandNumber =
        convOrDotOrBiasOut.use_begin()->getOperandNumber();

    if (auto scaleOp = dyn_cast_or_null<MulOp>(convOrDotOrBiasUser)) {
      auto scaleOperand = scaleOp->getOperand(1 - convOrDotOrBiasOperandNumber);
      auto maybeConstOpWithType = getConstOpWithType(scaleOperand, featureDim);
      if (!maybeConstOpWithType.has_value()) {
        return failure();
      }

      auto constOpWithType = *maybeConstOpWithType;
      mhlo::ConstantOp constOp = constOpWithType.first;
      if (constOpWithType.second != vectorWithBroadcast) {
        constOp = createVectorConstFromConst(featureDim, constOp, rewriter);
      }

      // Start to construct a new subgraph which could be const folded.
      if (!constOp->isBeforeInBlock(convOrDotOp))
        constOp->moveBefore(convOrDotOp);

      // construct new weight
      OpBuilder builder(convOrDotOp);
      auto weightType = cast<ShapedType>(weight.getType());
      BroadcastInDimOp newBroadInDimOp = builder.create<mhlo::BroadcastInDimOp>(
          constOp->getLoc(), weightType, constOp.getOutput(),
          rewriter.getI64TensorAttr({weightFeatureDim}));
      MulOp newMulOp = builder.create<MulOp>(constOp->getLoc(), weight,
                                             newBroadInDimOp->getResult(0));
      convOrDotOp->setOperand(1, newMulOp->getResult(0));

      // construct new bias
      if (biasAddOp) {
        OpBuilder builder(biasAddOp);
        auto constOpWithType =
            getConstOpWithType(biasOperand, featureDim).value();
        auto biasConstOp = constOpWithType.first;
        if (constOpWithType.second != vectorWithBroadcast) {
          biasConstOp =
              createVectorConstFromConst(featureDim, biasConstOp, rewriter);
        }
        MulOp newMulOp = builder.create<MulOp>(
            constOp->getLoc(), biasConstOp.getOutput(), constOp.getOutput());
        Value bcast = builder.create<mhlo::BroadcastInDimOp>(
            constOp->getLoc(), biasAddOp.getType(), newMulOp.getResult(),
            builder.getI64TensorAttr({featureDim}));
        biasAddOp->setOperand(biasConstOperandNumber, bcast);
      }

      // update uses
      scaleOp->getResult(0).replaceAllUsesWith(convOrDotOrBiasOut);

    } else if (auto offsetOp = dyn_cast_or_null<AddOp>(convOrDotOrBiasUser)) {
      if (!biasAddOp) {
        return failure();
      }
      auto offsetOpernd =
          offsetOp->getOperand(1 - convOrDotOrBiasOperandNumber);
      auto maybeConstOpWithType = getConstOpWithType(offsetOpernd, featureDim);
      if (!maybeConstOpWithType.has_value()) {
        return failure();
      }

      auto constOpWithType = *maybeConstOpWithType;
      mhlo::ConstantOp constOp = constOpWithType.first;
      if (constOpWithType.second != vectorWithBroadcast) {
        constOp = createVectorConstFromConst(featureDim, constOp, rewriter);
      }

      // Start to construct a new subgraph which could be const folded.
      if (!constOp->isBeforeInBlock(convOrDotOp))
        constOp->moveBefore(convOrDotOp);

      // construct new bias
      OpBuilder builder(biasAddOp);
      constOpWithType = *getConstOpWithType(biasOperand, featureDim);
      auto biasConstOp = constOpWithType.first;
      if (constOpWithType.second != vectorWithBroadcast) {
        biasConstOp =
            createVectorConstFromConst(featureDim, biasConstOp, rewriter);
      }
      mhlo::AddOp newAddOp = builder.create<AddOp>(
          constOp->getLoc(), biasConstOp.getOutput(), constOp.getOutput());
      Value bcast = builder.create<mhlo::BroadcastInDimOp>(
          constOp->getLoc(), biasAddOp.getType(), newAddOp.getResult(),
          builder.getI64TensorAttr({featureDim}));
      biasAddOp->setOperand(biasConstOperandNumber, bcast);

      // update conv's uses
      offsetOp->getResult(0).replaceAllUsesWith(convOrDotOrBiasOut);

    } else if (auto subOp = dyn_cast_or_null<SubtractOp>(convOrDotOrBiasUser)) {
      // conv_or_bias - a => conv_or_bias + (- a)

      // b_const should be rhs
      auto broadInDimOp =
          subOp.getRhs().getDefiningOp<mhlo::BroadcastInDimOp>();
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

    } else if (auto divOp = dyn_cast_or_null<DivOp>(convOrDotOrBiasUser)) {
      // conv_or_bias / a => conv_or_bias * (1 / a)

      // b_const should be rhs
      auto broadInDimOp =
          divOp.getRhs().getDefiningOp<mhlo::BroadcastInDimOp>();
      auto maybeConstOp = getBroadcastedConstOp(broadInDimOp, featureDim);
      if (!maybeConstOp.has_value())
        return failure();
      ConstantOp constOp = *maybeConstOp;

      OpBuilder builder(divOp);
      // replace b_const with 1 / b_const
      auto constType = cast<RankedTensorType>(constOp.getOutput().getType());
      auto fpType = dyn_cast<FloatType>(constType.getElementType());
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
    // handle a common, special case of ReduceWindow for 1 input, 1
    // init_values, and 1 result
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
               PadConvToConvPattern, 
               PadReduceWindowToReduceWindowPattern,
               ConvOrDotWithBiasFollowedByBroadcastPattern<mhlo::ConvolutionOp>,
               ConvOrDotWithBiasFollowedByBroadcastPattern<mhlo::DotOp>>(
          patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createHloFolderPass() {
  return std::make_unique<HloFolderPass>();
}
