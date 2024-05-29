//===- TensorToLinalg.cpp -------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/ToLinalg/ToLinalg.h"

#include "byteir/Utils/AffineUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;

namespace {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
static SmallVector<utils::IteratorType, 3>
getParallelAndReductionIterators(unsigned nLoops, unsigned nReduction) {
  SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
                                          utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);
  return res;
}

static SmallVector<utils::IteratorType, 3>
getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

static SmallVector<AffineMap, 4>
getLinearizedReassociationMaps(mlir::MLIRContext *ctx,
                               ArrayRef<AffineMap> reassoMaps,
                               ArrayRef<ReassociationIndices> reassoIndices,
                               ArrayRef<int64_t> staticShape) {
  SmallVector<AffineMap, 4> linearizedReassoMaps;
  for (auto zp : llvm::zip(reassoIndices, reassoMaps)) {
    SmallVector<int64_t> reassoShape;
    for (auto index : std::get<0>(zp)) {
      reassoShape.push_back(staticShape[index]);
    }
    auto map = getFlattenAffineMap(ctx, reassoShape);
    linearizedReassoMaps.push_back(map.compose(std::get<1>(zp)));
  }
  return linearizedReassoMaps;
}

class ExpandShapeToLinalgGeneric
    : public OpConversionPattern<tensor::ExpandShapeOp> {
public:
  using OpConversionPattern<tensor::ExpandShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExpandShapeOp op,
                  tensor::ExpandShapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ctx = op.getContext();
    auto resultTy = op.getResultType();
    int64_t nloops = resultTy.getRank();

    // Find input/output values and types.
    auto loc = op.getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                    resultTy.getElementType());
    Value output = emptyOp.getResult();
    auto outputShape = resultTy.getShape();

    SmallVector<AffineMap, 4> maps;

    maps.push_back(concatAffineMaps(getLinearizedReassociationMaps(
        ctx, op.getReassociationMaps(), op.getReassociationIndices(),
        outputShape)));
    maps.push_back(AffineMap::getMultiDimIdentityMap(nloops, ctx));

    // Build `linalg.generic` op.
    ValueRange inputs = adaptor.getOperands();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, inputs, output, maps, getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc,
                                                args.take_front(inputs.size()));
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

class CollapseShapeToLinalgGeneric
    : public OpConversionPattern<tensor::CollapseShapeOp> {
public:
  using OpConversionPattern<tensor::CollapseShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::CollapseShapeOp op,
                  tensor::CollapseShapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ctx = op.getContext();
    auto resultTy = op.getResultType();
    auto inputTy = op.getOperand().getType().cast<TensorType>();
    int64_t nloops = inputTy.getRank();

    // Find input/output values and types.
    auto loc = op.getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                    resultTy.getElementType());
    Value output = emptyOp.getResult();
    auto inputShape = inputTy.getShape();

    SmallVector<AffineMap, 4> maps;
    maps.push_back(AffineMap::getMultiDimIdentityMap(nloops, ctx));
    maps.push_back(concatAffineMaps(getLinearizedReassociationMaps(
        ctx, op.getReassociationMaps(), op.getReassociationIndices(),
        inputShape)));

    // Build `linalg.generic` op.
    ValueRange inputs = adaptor.getOperands();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, inputs, output, maps, getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc,
                                                args.take_front(inputs.size()));
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

class InsertSliceToLinalgGeneric
    : public OpConversionPattern<tensor::InsertSliceOp> {
public:
  using OpConversionPattern<tensor::InsertSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op,
                  tensor::InsertSliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto ctx = op.getContext();
    auto source = op.getSource();
    auto dest = op.getDest();
    auto resultTy = op.getResultType();
    auto inputTy = op.getSourceType();
    int64_t nloops = resultTy.getRank();

    if (ShapedType::isDynamicShape(resultTy.getShape())) {
      return failure();
    }

    // Find input/output values and types.
    auto loc = op.getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                    resultTy.getElementType());
    Value output = emptyOp.getResult();

    SmallVector<AffineMap> maps;
    maps.push_back(AffineMap::getMultiDimIdentityMap(nloops, ctx));

    // Build `linalg.generic` op.
    ValueRange inputs = adaptor.getOperands();
    auto ranges = op.getOrCreateRanges(rewriter, loc);
    llvm::SmallBitVector droppedDims = op.getDroppedDims();

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, ValueRange(), output, maps,
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          SmallVector<Value> linalgIndices;
          for (int64_t i = 0; i < nloops; ++i) {
            linalgIndices.emplace_back(
                nestedBuilder.create<linalg::IndexOp>(loc, i));
          }

          SmallVector<Value> srcIndices;
          Value predicate;
          int64_t srcDimIdx = 0;
          for (int64_t i = 0; i < resultTy.getRank(); ++i) {
            if (droppedDims.test(i))
              continue;
            srcIndices.emplace_back(linalgIndices[i]);

            AffineExpr offset, stride, idx;
            SmallVector<OpFoldResult> symbolVals = {
                linalgIndices[i],
                ranges[i].offset,
                ranges[i].stride,
            };

            bindSymbols(nestedBuilder.getContext(), idx, offset, stride);
            OpFoldResult remainder = affine::makeComposedFoldedAffineApply(
                nestedBuilder, loc,
                AffineMap::get(0, 3, (idx - offset) % stride), symbolVals);

            Value remainderVal =
                getValueOrCreateConstantIndexOp(nestedBuilder, loc, remainder);

            OpFoldResult division = affine::makeComposedFoldedAffineApply(
                nestedBuilder, loc,
                AffineMap::get(0, 3, (idx - offset).floorDiv(stride)),
                symbolVals);
            Value divisionVal =
                getValueOrCreateConstantIndexOp(nestedBuilder, loc, division);

            Value zero = nestedBuilder.create<arith::ConstantIndexOp>(loc, 0);
            Value equalZero = nestedBuilder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, remainderVal, zero);

            Value size = getValueOrCreateConstantIndexOp(nestedBuilder, loc,
                                                         ranges[i].size);
            Value inBound = nestedBuilder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::ult, divisionVal, size);
            Value curPredicate =
                nestedBuilder.create<arith::AndIOp>(loc, equalZero, inBound);

            predicate = predicate ? nestedBuilder.create<arith::AndIOp>(
                                        loc, predicate, curPredicate)
                                  : curPredicate;
            srcDimIdx += 1;
          }

          auto ifPred = nestedBuilder.create<scf::IfOp>(
              loc, resultTy.getElementType(), predicate,
              /*withElseRegion=*/true);

          // Pred == true, return source
          {
            OpBuilder ifPredThenB = ifPred.getThenBodyBuilder();
            Value val =
                ifPredThenB.create<tensor::ExtractOp>(loc, source, srcIndices);
            ifPredThenB.create<scf::YieldOp>(loc, val);
          }

          // Pred == false, therefore return dest.
          {
            OpBuilder ifPredElseB = ifPred.getElseBodyBuilder();
            Value val =
                ifPredElseB.create<tensor::ExtractOp>(loc, dest, linalgIndices);
            ifPredElseB.create<scf::YieldOp>(loc, val);
          }
          nestedBuilder.create<linalg::YieldOp>(loc, ifPred.getResults());
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct TensorToLinalgPass : public TensorToLinalgBase<TensorToLinalgPass> {

  TensorToLinalgPass() = default;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<cf::ControlFlowDialect, func::FuncDialect,
                    linalg::LinalgDialect, scf::SCFDialect, math::MathDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);

    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, linalg::LinalgDialect,
                           math::MathDialect, scf::SCFDialect,
                           shape::ShapeDialect, affine::AffineDialect>();
    target.addLegalOp<tensor::EmptyOp, tensor::ExtractOp>();

    target.addDynamicallyLegalOp<tensor::ExpandShapeOp>(
        [&](tensor::ExpandShapeOp op) {
          return !op.getResultType().hasStaticShape();
        });
    target.addDynamicallyLegalOp<tensor::CollapseShapeOp>(
        [&](tensor::CollapseShapeOp op) {
          return !op.getResultType().hasStaticShape();
        });
    target.addDynamicallyLegalOp<tensor::ExtractSliceOp>(
        [&](tensor::ExtractSliceOp op) {
          return !op.getResultType().hasStaticShape();
        });
    populateTensorToLinalgConversionPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

LogicalResult mlir::simplifyTensorReshapeLikeOp(RewriterBase &rewriter,
                                                Operation *op) {
  auto ctx = op->getContext();
  auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
  if (!extractSliceOp) {
    return failure();
  }

  auto src = extractSliceOp.getSource();
  auto loc = op->getLoc();
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedOffsets =
      extractSliceOp.getMixedOffsets();
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes =
      extractSliceOp.getMixedSizes();
  rewriter.setInsertionPoint(extractSliceOp);
  if (auto expandShapeOp = src.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto expandResultTy = expandShapeOp.getResultType();
    auto expandOutputShape = expandResultTy.getShape();
  } else if (auto collapseShapeOp =
                 src.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto collpaseInputTy =
        collapseShapeOp.getOperand().getType().cast<TensorType>();
    auto collpaseInputShape = collpaseInputTy.getShape();
    auto maps = getLinearizedReassociationMaps(
        ctx, collapseShapeOp.getReassociationMaps(),
        collapseShapeOp.getReassociationIndices(), collpaseInputTy.getShape());
  }

  return success();
}

void mlir::populateTensorToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExpandShapeToLinalgGeneric, CollapseShapeToLinalgGeneric,
               InsertSliceToLinalgGeneric>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createTensorToLinalgPass() {
  return std::make_unique<TensorToLinalgPass>();
}
