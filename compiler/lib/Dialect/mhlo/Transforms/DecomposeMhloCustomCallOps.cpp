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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

struct DecomposeByteIRSoftmax : public OpRewritePattern<mhlo::CustomCallOp> {
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

    RankedTensorType inType =
        cast<RankedTensorType>(op.getOperand(0).getType());
    Value exp = rewriter.create<mhlo::ExpOp>(op.getLoc(), op.getOperand(0));
    Value reduce;
    {
      SmallVector<int64_t> reduceResultShape(inType.getShape());
      reduceResultShape.erase(reduceResultShape.begin() + axisAttr.getInt());
      RankedTensorType reduceResultType =
          RankedTensorType::get(reduceResultShape, inType.getElementType());

      Value initValue = rewriter.create<mhlo::ConstantOp>(
          op.getLoc(),
          DenseElementsAttr::get(
              RankedTensorType::get({}, inType.getElementType()),
              {APFloat::getZero(cast<mlir::FloatType>(inType.getElementType())
                                    .getFloatSemantics())}));
      auto reduceOp = rewriter.create<mhlo::ReduceOp>(
          op.getLoc(), reduceResultType, exp, initValue,
          rewriter.getI64TensorAttr({axisAttr.getInt()}));

      Block &block = reduceOp.getBody().emplaceBlock();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      auto blockValArgumentType =
          RankedTensorType::get({}, inType.getElementType());
      block.addArgument(blockValArgumentType, op->getLoc());
      block.addArgument(blockValArgumentType, op->getLoc());
      auto *firstValArg = block.args_begin();
      auto *secondValArg = std::next(firstValArg);
      Value result = rewriter.create<mhlo::AddOp>(op->getLoc(), *firstValArg,
                                                  *secondValArg);
      rewriter.create<mhlo::ReturnOp>(op->getLoc(), result);

      reduce = reduceOp.getResults()[0];
    }

    SmallVector broadcastDim =
        llvm::to_vector(llvm::seq<int64_t>(0, inType.getRank()));
    broadcastDim.erase(broadcastDim.begin() + axisAttr.getInt());
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        op->getLoc(), inType, reduce,
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), exp),
        rewriter.getI64TensorAttr(broadcastDim));
    Value result = rewriter.create<mhlo::DivOp>(op->getLoc(), exp, broadcast);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct DecomposeByteIRL2Norm : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != getL2NormName()) {
      return failure();
    }

    Value operand = op.getOperand(0);
    RankedTensorType inType = cast<RankedTensorType>(operand.getType());
    mlir::FloatType fpType = cast<FloatType>(inType.getElementType());

    DictionaryAttr byteirAttrs =
        cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
    if (!byteirAttrs)
      return failure();
    auto axisAttr = cast<ArrayAttr>(byteirAttrs.get("axis"));
    if (axisAttr.size() != 1) {
      return op->emitError("only support 1 axis");
    }
    auto axis = cast<IntegerAttr>(axisAttr[0]).getInt();

    auto epsAttr = cast<FloatAttr>(byteirAttrs.get("epsilon"));
    APFloat eps = epsAttr.getValue();
    bool losesInfo;
    auto status = eps.convert(fpType.getFloatSemantics(),
                              APFloat::rmNearestTiesToEven, &losesInfo);
    if (losesInfo) {
      op->emitRemark("loses info when eps convert to input type");
    }
    epsAttr = rewriter.getFloatAttr(fpType, eps);

    bool epsOutsideSqrt = false;
    if (byteirAttrs.contains("eps_outside_sqrt")) {
      epsOutsideSqrt =
          cast<BoolAttr>(byteirAttrs.get("eps_outside_sqrt")).getValue();
    }

    Value pow2 = rewriter.create<mhlo::MulOp>(op.getLoc(), operand, operand);
    Value reduce;
    {
      SmallVector<int64_t> reduceResultShape(inType.getShape());
      reduceResultShape.erase(reduceResultShape.begin() + axis);
      RankedTensorType reduceResultType =
          RankedTensorType::get(reduceResultShape, fpType);

      Value initValue = rewriter.create<mhlo::ConstantOp>(
          op.getLoc(), DenseElementsAttr::get(
                           RankedTensorType::get({}, fpType),
                           {APFloat::getZero(fpType.getFloatSemantics())}));
      auto reduceOp = rewriter.create<mhlo::ReduceOp>(
          op.getLoc(), reduceResultType, pow2, initValue,
          rewriter.getI64TensorAttr({axis}));

      Block &block = reduceOp.getBody().emplaceBlock();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      auto blockValArgumentType =
          RankedTensorType::get({}, inType.getElementType());
      block.addArgument(blockValArgumentType, op->getLoc());
      block.addArgument(blockValArgumentType, op->getLoc());
      auto *firstValArg = block.args_begin();
      auto *secondValArg = std::next(firstValArg);
      Value result = rewriter.create<mhlo::AddOp>(op->getLoc(), *firstValArg,
                                                  *secondValArg);
      rewriter.create<mhlo::ReturnOp>(op->getLoc(), result);

      reduce = reduceOp.getResults()[0];
    }

    Value epsValue = rewriter.create<mhlo::ConstantOp>(
        op.getLoc(),
        DenseElementsAttr::get(RankedTensorType::get({}, fpType), epsAttr));
    epsValue = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        op.getLoc(), reduce.getType(), epsValue,
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), reduce),
        rewriter.getI64TensorAttr({}));
    Value sqrt;
    if (epsOutsideSqrt) {
      sqrt = rewriter.create<mhlo::SqrtOp>(op.getLoc(), reduce);
      sqrt = rewriter.create<mhlo::MaxOp>(op.getLoc(), sqrt, epsValue);
    } else {
      sqrt = rewriter.create<mhlo::MaxOp>(op.getLoc(), reduce, epsValue);
      sqrt = rewriter.create<mhlo::SqrtOp>(op.getLoc(), sqrt);
    }

    SmallVector broadcastDim =
        llvm::to_vector(llvm::seq<int64_t>(0, inType.getRank()));
    broadcastDim.erase(broadcastDim.begin() + axis);
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        op->getLoc(), inType, sqrt,
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), operand),
        rewriter.getI64TensorAttr(broadcastDim));

    Value result =
        rewriter.create<mhlo::DivOp>(op->getLoc(), operand, broadcast);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct DecomposeByteIRArgMaxMin : public OpRewritePattern<mhlo::CustomCallOp> {
  DecomposeByteIRArgMaxMin(MLIRContext *context, llvm::StringRef customCallName)
      : OpRewritePattern<mhlo::CustomCallOp>(context),
        customCallName(customCallName.str()) {}
  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != customCallName)
      return failure();

    DictionaryAttr byteirAttrs =
        cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
    if (!byteirAttrs)
      return failure();
    auto axisAttr = cast<IntegerAttr>(byteirAttrs.get("axis"));
    auto keepDimAttr = cast<BoolAttr>(byteirAttrs.get("keep_dims"));
    auto selectLastIndexAttr =
        cast<BoolAttr>(byteirAttrs.get("select_last_index"));
    if (selectLastIndexAttr.getValue()) {
      return op.emitError("unimplemented: select_last_index = true");
    }
    // TODO(lyq): support keep_dims = true
    if (keepDimAttr.getValue()) {
      return op.emitError("unimplemented: keep_dims = true");
    }

    RankedTensorType inType =
        cast<RankedTensorType>(op.getOperand(0).getType());
    Type inElemType = inType.getElementType();
    RankedTensorType outType, outIndexType;
    if (op.getResults().size() == 1) {
      outIndexType = cast<RankedTensorType>(op.getResults()[0].getType());
      outType = outIndexType.clone(inElemType);
    } else if (op.getResults().size() == 2) {
      outType = cast<RankedTensorType>(op.getResults()[0].getType());
      outIndexType = cast<RankedTensorType>(op.getResults()[1].getType());
    } else {
      return op.emitError("unsupported result size");
    }

    if (!isa<mlir::FloatType, mlir::IntegerType>(inElemType)) {
      return op.emitError("only support float or int type");
    }

    // create init values
    Value initValue;
    if (customCallName == getArgMaxName().str()) {
      if (isa<mlir::FloatType>(inElemType)) {
        initValue = rewriter.create<mhlo::ConstantOp>(
            op.getLoc(),
            DenseElementsAttr::get(
                RankedTensorType::get({}, inElemType),
                {APFloat::getInf(
                    cast<mlir::FloatType>(inElemType).getFloatSemantics(),
                    /*negative=*/true)}));
      } else if (isa<mlir::IntegerType>(inElemType)) {
        if (cast<mlir::IntegerType>(inElemType).isSignless() &&
            inElemType.getIntOrFloatBitWidth() != 1) {
          initValue = rewriter.create<mhlo::ConstantOp>(
              op.getLoc(),
              DenseElementsAttr::get(RankedTensorType::get({}, inElemType),
                                     {APInt::getSignedMinValue(
                                         inElemType.getIntOrFloatBitWidth())}));
        } else {
          initValue = rewriter.create<mhlo::ConstantOp>(
              op.getLoc(),
              DenseElementsAttr::get(
                  RankedTensorType::get({}, inElemType),
                  {APInt::getMinValue(inElemType.getIntOrFloatBitWidth())}));
        }
      }
    } else if (customCallName == getArgMinName().str()) {
      if (isa<mlir::FloatType>(inElemType)) {
        initValue = rewriter.create<mhlo::ConstantOp>(
            op.getLoc(),
            DenseElementsAttr::get(
                RankedTensorType::get({}, inElemType),
                {APFloat::getInf(
                    cast<mlir::FloatType>(inElemType).getFloatSemantics(),
                    /*negative=*/false)}));
      } else if (isa<mlir::IntegerType>(inElemType)) {
        if (cast<mlir::IntegerType>(inElemType).isSignless() &&
            inElemType.getIntOrFloatBitWidth() != 1) {
          initValue = rewriter.create<mhlo::ConstantOp>(
              op.getLoc(),
              DenseElementsAttr::get(RankedTensorType::get({}, inElemType),
                                     {APInt::getSignedMaxValue(
                                         inElemType.getIntOrFloatBitWidth())}));
        } else {
          initValue = rewriter.create<mhlo::ConstantOp>(
              op.getLoc(),
              DenseElementsAttr::get(
                  RankedTensorType::get({}, inElemType),
                  {APInt::getMaxValue(inElemType.getIntOrFloatBitWidth())}));
        }
      }
    } else {
      return op.emitError("unknown custom call name");
    }
    Value initIndex = rewriter.create<mhlo::ConstantOp>(
        op.getLoc(),
        DenseElementsAttr::get(
            RankedTensorType::get({}, outIndexType.getElementType()),
            {APInt::getZero(
                outIndexType.getElementType().getIntOrFloatBitWidth())}));

    llvm::SmallVector<Value> inputShapeVec;
    for (int64_t i = 0; i < inType.getRank(); i++) {
      inputShapeVec.push_back(rewriter.create<tensor::DimOp>(
          op.getLoc(), op.getOperand(0),
          rewriter.create<arith::ConstantOp>(op.getLoc(),
                                             rewriter.getIndexAttr(i))));
    }
    Value inputShapeTensor =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), inputShapeVec);
    Value indexTensor = rewriter.create<mhlo::DynamicIotaOp>(
        op.getLoc(), inType.clone(outIndexType.getElementType()),
        inputShapeTensor, axisAttr);
    auto reduceOp = rewriter.create<mhlo::ReduceOp>(
        op.getLoc(), TypeRange{outType, outIndexType},
        ValueRange{op.getOperand(0), indexTensor},
        ValueRange{initValue, initIndex},
        rewriter.getI64TensorAttr({axisAttr.getInt()}));
    {
      Block &block = reduceOp.getBody().emplaceBlock();
      // Add block arguments
      auto blockValArgumentType = RankedTensorType::get({}, inElemType);
      auto blockIdxArgumentType =
          RankedTensorType::get({}, outIndexType.getElementType());
      auto compareResultType = RankedTensorType::get({}, rewriter.getI1Type());
      block.addArgument(blockValArgumentType, op->getLoc());
      block.addArgument(blockIdxArgumentType, op->getLoc());

      block.addArgument(blockValArgumentType, op->getLoc());
      block.addArgument(blockIdxArgumentType, op->getLoc());

      auto *firstValArg = block.args_begin();
      auto *firstIdxArg = std::next(firstValArg);
      auto *secondValArg = std::next(firstIdxArg);
      auto *secondIdxArg = std::next(secondValArg);

      mhlo::ComparisonTypeAttr compareTypeAttr = mhlo::ComparisonTypeAttr::get(
          rewriter.getContext(), mhlo::ComparisonType::FLOAT);
      if (isa<mlir::IntegerType>(inElemType)) {
        if (cast<mlir::IntegerType>(inElemType).isSignless() &&
            inElemType.getIntOrFloatBitWidth() != 1) {
          compareTypeAttr = mhlo::ComparisonTypeAttr::get(
              rewriter.getContext(), mhlo::ComparisonType::SIGNED);
        } else {
          compareTypeAttr = mhlo::ComparisonTypeAttr::get(
              rewriter.getContext(), mhlo::ComparisonType::UNSIGNED);
        }
      }
      mhlo::ComparisonDirectionAttr compareGeDirectionAttr =
          mhlo::ComparisonDirectionAttr::get(rewriter.getContext(),
                                             mhlo::ComparisonDirection::GE);
      mhlo::ComparisonDirectionAttr compareLeDirectionAttr =
          mhlo::ComparisonDirectionAttr::get(rewriter.getContext(),
                                             mhlo::ComparisonDirection::LE);
      mhlo::ComparisonDirectionAttr compareEqDirectionAttr =
          mhlo::ComparisonDirectionAttr::get(rewriter.getContext(),
                                             mhlo::ComparisonDirection::EQ);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      Value compareResult;
      if (customCallName == getArgMaxName().str()) {
        compareResult = rewriter.create<mhlo::CompareOp>(
            op->getLoc(), compareResultType, *firstValArg, *secondValArg,
            compareGeDirectionAttr, compareTypeAttr);
      } else {
        compareResult = rewriter.create<mhlo::CompareOp>(
            op->getLoc(), compareResultType, *firstValArg, *secondValArg,
            compareLeDirectionAttr, compareTypeAttr);
      }

      Value retValResult = rewriter.create<mhlo::SelectOp>(
          op->getLoc(), compareResult, *firstValArg, *secondValArg);

      // get smaller index value if compared nums are equal.
      Value compareEqResult = rewriter.create<mhlo::CompareOp>(
          op->getLoc(), compareResultType, *firstValArg, *secondValArg,
          compareEqDirectionAttr, compareTypeAttr);
      Value minIdx = rewriter.create<mhlo::MinOp>(op->getLoc(), *firstIdxArg,
                                                  *secondIdxArg);
      Value idxWithGeVal = rewriter.create<mhlo::SelectOp>(
          op->getLoc(), compareResult, *firstIdxArg, *secondIdxArg);
      Value retIdxResult = rewriter.create<mhlo::SelectOp>(
          op->getLoc(), compareEqResult, minIdx, idxWithGeVal);

      rewriter.create<mhlo::ReturnOp>(op->getLoc(),
                                      ValueRange{retValResult, retIdxResult});
    }

    if (op.getResults().size() == 1) {
      rewriter.replaceOp(op, reduceOp.getResults()[1]);
    } else {
      rewriter.replaceOp(op, reduceOp.getResults());
    }
    return success();
  }

  std::string customCallName;
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
    if (!legalOpsSet.contains(getSoftmaxName())) {
      patterns.add<DecomposeByteIRSoftmax>(context);
    }
    if (!legalOpsSet.contains(getL2NormName())) {
      patterns.add<DecomposeByteIRL2Norm>(context);
    }
    if (!legalOpsSet.contains(getArgMaxName())) {
      patterns.add<DecomposeByteIRArgMaxMin>(context, getArgMaxName());
    }
    if (!legalOpsSet.contains(getArgMinName())) {
      patterns.add<DecomposeByteIRArgMaxMin>(context, getArgMinName());
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
