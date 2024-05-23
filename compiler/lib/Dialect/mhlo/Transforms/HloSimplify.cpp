//===- HloSimplify.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/HloSimplify.h"

#include "PassDetail.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

static Value createIntialZeroValue(Type elementTy, PatternRewriter &rewriter,
                                   Location &loc) {
  auto constType = RankedTensorType::get({}, elementTy);
  if (isa<mlir::FloatType>(elementTy)) {
    auto constAttr = DenseElementsAttr::get(
        constType,
        {APFloat::getZero(cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                          /*negative=*/false)});
    return rewriter.create<mhlo::ConstantOp>(loc, constType, constAttr);
  } else if (isa<mlir::IntegerType>(elementTy) &&
             elementTy.getIntOrFloatBitWidth() != 8) {
    auto constAttr = DenseElementsAttr::get(
        constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
    return rewriter.create<mhlo::ConstantOp>(loc, constType, constAttr);
  }

  return nullptr;
}

struct SimplifyReduceToReshape : public OpRewritePattern<mhlo::ReduceOp> {
  using OpRewritePattern<mhlo::ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    bool isRegular = isRegularReduceOp<mhlo::AddOp>(op) ||
                     isRegularReduceOp<mhlo::MaxOp>(op) ||
                     isRegularReduceOp<mhlo::MinOp>(op) ||
                     isRegularReduceOp<mhlo::OrOp>(op) ||
                     isRegularReduceOp<mhlo::MulOp>(op);
    if (!isRegular) {
      return failure();
    }

    Value input = op.getInputs()[0];
    Value output = op.getResults()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!inputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape()) {
      // TODO: support dynamic shape
      return failure();
    }
    auto dimensions = op.getDimensions().getValues<int64_t>();
    for (int64_t i : dimensions) {
      if (inputType.getDimSize(i) != 1) {
        return failure();
      }
    }

    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, outputType, input);
    return success();
  }
};

// slice(broadcast_in_dim(x)) => broadcast_in_dim(x)
struct SimplifySliceBroadcastToBroadcast
    : public OpRewritePattern<mhlo::SliceOp> {
  using OpRewritePattern<mhlo::SliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    if (auto bcastOp =
            op.getOperand().getDefiningOp<mhlo::BroadcastInDimOp>()) {
      auto operandTy = cast<RankedTensorType>(bcastOp.getOperand().getType());
      if (operandTy.getRank() == 0) {
        rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
            op, op.getType(), bcastOp.getOperand(),
            bcastOp.getBroadcastDimensions());
        return success();
      }
    }
    return failure();
  }
};

struct SimplifyDotGeneralToBroadcastMultiply
    : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());

    // TODO: support dynamic shape
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
      return failure();
    }

    // only support [b, m, k] @ [b, k, n] or [m, k] @ [k, n]
    auto dimensionNumbers = op.getDotDimensionNumbers();
    if (dimensionNumbers.getLhsContractingDimensions().size() != 1 ||
        dimensionNumbers.getRhsContractingDimensions().size() != 1) {
      return failure();
    }
    if (dimensionNumbers.getLhsBatchingDimensions().size() == 0 &&
        dimensionNumbers.getRhsBatchingDimensions().size() == 0) {
      if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
        return failure();
      }
    } else if (dimensionNumbers.getLhsBatchingDimensions().size() == 1 &&
               dimensionNumbers.getRhsBatchingDimensions().size() == 1) {
      if (lhsType.getRank() != 3 || rhsType.getRank() != 3) {
        return failure();
      }
      if (dimensionNumbers.getLhsBatchingDimensions()[0] != 0 ||
          dimensionNumbers.getRhsBatchingDimensions()[0] != 0) {
        return failure();
      }
    } else {
      return failure();
    }

    if (dimensionNumbers.getLhsContractingDimensions()[0] !=
            lhsType.getRank() - 1 ||
        dimensionNumbers.getRhsContractingDimensions()[0] !=
            rhsType.getRank() - 2) {
      return failure();
    }

    auto is_single_k = [&]() {
      return lhsType.getDimSize(
                 dimensionNumbers.getLhsContractingDimensions()[0]) == 1 &&
             rhsType.getDimSize(
                 dimensionNumbers.getRhsContractingDimensions()[0]) == 1;
    };

    auto is_single_m_n = [&]() {
      return lhsType.getDimSize(lhsType.getRank() - 2) == 1 ||
             rhsType.getDimSize(rhsType.getRank() - 1) == 1;
    };

    auto resultRank = op.getType().getRank();
    auto resultShape = op.getType().getShape();
    auto resultElementTy = op.getType().getElementType();
    auto has_batch = dimensionNumbers.getLhsBatchingDimensions().size() != 0 ||
                     dimensionNumbers.getRhsBatchingDimensions().size();
    auto B = has_batch ? resultShape[0] : 0;
    auto M = lhsType.getDimSize(lhsType.getRank() - 2);
    auto N = rhsType.getDimSize(rhsType.getRank() - 1);
    auto LK =
        lhsType.getDimSize(dimensionNumbers.getLhsContractingDimensions()[0]);
    auto RK =
        rhsType.getDimSize(dimensionNumbers.getRhsContractingDimensions()[0]);
    auto K = std::max(LK, RK);

    if (is_single_k()) {
      // k == 1
      auto iota = llvm::iota_range<int64_t>(0, lhsType.getRank(), false);
      llvm::SmallVector<int64_t> bcastDims(iota.begin(), iota.end());
      Value bcastLhs = rewriter.create<mhlo::BroadcastInDimOp>(
          loc, op.getType(), op.getLhs(), rewriter.getI64TensorAttr(bcastDims));
      Value bcastRhs = rewriter.create<mhlo::BroadcastInDimOp>(
          loc, op.getType(), op.getRhs(), rewriter.getI64TensorAttr(bcastDims));
      rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, op.getType(), bcastLhs,
                                               bcastRhs);
      return success();
    } else if (is_single_m_n()) {
      // m == 1 or n == 1
      const bool is_single_m = M == 1;
      auto lhsRank = lhsType.getRank();
      auto rhsRank = rhsType.getRank();

      // broadcast lhs and rhs.
      auto lIota = llvm::iota_range<int64_t>(0, lhsRank, false);
      auto rIota = llvm::iota_range<int64_t>(0, rhsRank, false);
      llvm::SmallVector<int64_t, 3> lhsBcastDims(lIota.begin(), lIota.end());
      llvm::SmallVector<int64_t, 3> rhsBcastDims(rIota.begin(), rIota.end());
      if (is_single_m)
        std::swap(lhsBcastDims[lhsRank - 1], lhsBcastDims[lhsRank - 2]);
      else
        std::swap(rhsBcastDims[rhsRank - 1], rhsBcastDims[rhsRank - 2]);
      // result shape of mulop is: [B,K,N] or [B,M,K].
      llvm::SmallVector<int64_t, 3> mulShape(resultShape);
      if (is_single_m)
        mulShape[resultRank - 2] = K;
      else
        mulShape[resultRank - 1] = K;
      auto mulType = RankedTensorType::get(mulShape, resultElementTy);
      Value bcastLhs = rewriter.create<mhlo::BroadcastInDimOp>(
          loc, mulType, op.getLhs(), rewriter.getI64TensorAttr(lhsBcastDims));
      Value bcastRhs = rewriter.create<mhlo::BroadcastInDimOp>(
          loc, mulType, op.getRhs(), rewriter.getI64TensorAttr(rhsBcastDims));

      // replace bmm with mul and reduce.
      Value mulOp =
          rewriter.create<mhlo::MulOp>(loc, mulType, bcastLhs, bcastRhs);
      llvm::SmallVector<int64_t, 2> reduceShape;
      if (has_batch)
        reduceShape.push_back(B);
      if (is_single_m)
        reduceShape.push_back(N);
      else
        reduceShape.push_back(M);
      auto initVal = createIntialZeroValue(resultElementTy, rewriter, loc);
      auto dim = is_single_m ? resultRank - 2 : resultRank - 1;
      auto reduceOp = rewriter.create<mhlo::ReduceOp>(
          loc, TypeRange{RankedTensorType::get(reduceShape, resultElementTy)},
          ValueRange{mulOp}, ValueRange{initVal},
          rewriter.getI64VectorAttr({dim}));

      Block &block = reduceOp.getBody().emplaceBlock();
      auto blockValArgType = RankedTensorType::get({}, resultElementTy);
      block.addArgument(blockValArgType, loc);
      block.addArgument(blockValArgType, loc);
      auto *firstArg = block.args_begin();
      auto *secondArg = std::next(firstArg);
      // create reduceOp body.
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&block);
        auto addOp = rewriter.create<mhlo::AddOp>(loc, *firstArg, *secondArg);
        rewriter.create<mhlo::ReturnOp>(loc, mlir::ValueRange{addOp});
      }

      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, op.getType(),
                                                   reduceOp.getResults());
      return success();
    }

    return failure();
  }
};

// reshape gather indices to 1D
struct ReshapeGatherPattern : public OpRewritePattern<mhlo::GatherOp> {
  using OpRewritePattern<mhlo::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::GatherOp op,
                                PatternRewriter &rewriter) const override {
    // for >1D indices:
    // gather(tensor, indices) => reshape(gather(tensor,reshape(indices)))
    auto startIndices = op.getStartIndices();
    auto startIndicesTy = startIndices.getType().cast<ShapedType>();
    if (!startIndicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked start_indices");
    }

    auto operand = op.getOperand();
    auto operandTy = operand.getType().cast<ShapedType>();
    if (!operandTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked operand");
    }

    int64_t indexVectorDim = startIndicesTy.getRank();

    auto dimensionNumbers = op.getDimensionNumbers();
    if (dimensionNumbers.getIndexVectorDim() != indexVectorDim) {
      return rewriter.notifyMatchFailure(
          op, "index_vector_dim not last dimension of start_indices");
    }

    // Index select only works across a single dimension.
    if (startIndicesTy.getShape().empty()) {
      return rewriter.notifyMatchFailure(
          op, "empty start_indices index vector dimension");
    }

    // Only support the default case for start_index_map.
    if (dimensionNumbers.getStartIndexMap().size() != 1 ||
        dimensionNumbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(op, "start_index_map != [0]");
    }

    auto resultTy = op.getResult().getType().dyn_cast<ShapedType>();
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "unranked result");
    }

    // Offset dimensions should be the defaults.
    if (dimensionNumbers.getOffsetDims().size() !=
        static_cast<size_t>(resultTy.getRank() - indexVectorDim)) {
      return rewriter.notifyMatchFailure(
          op, "offset_dims.size not operand rank minus index_vector_dim");
    }

    for (auto it : llvm::enumerate(dimensionNumbers.getOffsetDims())) {
      if ((it.index() + indexVectorDim) != static_cast<size_t>(it.value())) {
        return rewriter.notifyMatchFailure(
            op, "offset_dims != [index_vector_dim, result.rank)");
      }
    }

    for (auto it : llvm::enumerate(op.getSliceSizes().getValues<APInt>())) {
      // First shape value must be 1.
      if (it.index() == 0) {
        if (it.value().getSExtValue() != 1) {
          return rewriter.notifyMatchFailure(op, "slice_size[0] != 1");
        }
        continue;
      }

      // The op needs to index the entire slice for each other dimension.
      if (it.value().getSExtValue() != operandTy.getDimSize(it.index())) {
        return rewriter.notifyMatchFailure(
            op, "slice_size doesn't match operand dimension");
      }
    }

    if (dimensionNumbers.getCollapsedSliceDims().size() != 1 ||
        dimensionNumbers.getCollapsedSliceDims()[0] != 0) {
      return rewriter.notifyMatchFailure(op, "collapsed_slice_dims != [0]");
    }

    if (startIndicesTy.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "already 1D indices");

    auto indicesShape = startIndicesTy.getShape();
    int64_t numel = 1;
    for (auto dim : indicesShape)
      numel *= dim;
    RankedTensorType indices_reshaped_type =
        RankedTensorType::get({numel}, startIndicesTy.getElementType());
    auto indicesReshapeOp = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), indices_reshaped_type, startIndices);
    int64_t indexVecDim = 1;

    SmallVector<int64_t> offsetDims = {};
    SmallVector<int64_t> startIndexMap = {0};
    SmallVector<int64_t> collapsedDims = {0};
    for (auto it : llvm::enumerate(dimensionNumbers.getOffsetDims()))
      offsetDims.push_back(indexVecDim + it.index());

    auto dimsAttr = mhlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offsetDims=*/offsetDims,
        /*collapsedSliceDims=*/collapsedDims,
        /*startIndexMap=*/startIndexMap,
        /*indexVecDim=*/indexVecDim);
    auto gatherOp = rewriter.create<mhlo::GatherOp>(
        op.getLoc(), operand, indicesReshapeOp.getResult(), dimsAttr,
        op.getSliceSizes(), op.getIndicesAreSortedAttr());
    auto gatherReshapeOp =
        rewriter.create<mhlo::ReshapeOp>(op.getLoc(), resultTy, gatherOp);
    rewriter.replaceOp(op, gatherReshapeOp.getResult());
    return success();
  }
};

} // namespace

namespace {
struct HloSimplifyPass : public HloSimplifyBase<HloSimplifyPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<SimplifyReduceToReshape>(context);
    patterns.add<SimplifySliceBroadcastToBroadcast>(context);
    patterns.add<SimplifyDotGeneralToBroadcastMultiply>(context);
    patterns.add<ReshapeGatherPattern>(context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createHloSimplifyPass() {
  return std::make_unique<HloSimplifyPass>();
}
