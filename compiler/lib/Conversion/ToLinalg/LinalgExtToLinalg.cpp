//===- LinalgExtToLinalg.cpp ----------------------------------*--- C++ -*-===//
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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Utils/AffineUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
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

template <typename T>
static Value createReduce(Value input, Value init, int64_t reductionAxis,
                          Location loc, OpBuilder &builder) {

  SmallVector<Value> inputs{input};
  SmallVector<Value> inits{init};
  SmallVector<int64_t> reductionDims{reductionAxis};
  auto reduceOp = builder.create<linalg::ReduceOp>(
      loc, inputs, inits, reductionDims,
      [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        auto innerOp = nestedBuilder.create<T>(nestedLoc, args);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, innerOp->getResults());
      });
  return reduceOp.getResult(0);
}

struct SoftmaxOpToLinalgReduceAndGeneric
    : public OpConversionPattern<linalg_ext::SoftmaxOp> {
public:
  using OpConversionPattern<linalg_ext::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg_ext::SoftmaxOp op,
                  linalg_ext::SoftmaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // check op has tensor semantics here

    auto ctx = op.getContext();
    // Find input/output values and types.
    auto loc = op.getLoc();

    // 1. Build Max
    auto partialMax = createReduce<arith::MaxFOp>(
        op.input(), op.max(), op.getDimension(), loc, rewriter);

    auto resultTy = op.getType(0).cast<RankedTensorType>();
    // Create an empty
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                    resultTy.getElementType());

    int64_t nloops = resultTy.getRank();
    // calculate maps with broadcast
    SmallVector<AffineMap, 4> mapWithBroadcast;
    // input
    mapWithBroadcast.push_back(AffineMap::getMultiDimIdentityMap(nloops, ctx));
    // reduced input, e.g. max or acc
    mapWithBroadcast.push_back(getMultiDimIdentityMapWithSkips(
        nloops, {static_cast<int64_t>(op.getDimension())}, ctx));
    // init
    mapWithBroadcast.push_back(AffineMap::getMultiDimIdentityMap(nloops, ctx));

    // 2. Build exp(x-max)
    SmallVector<Value> expInputs{op.input(), partialMax};
    auto partialExpOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, expInputs, emptyOp->getResults(), mapWithBroadcast,
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto subOp =
              nestedBuilder.create<arith::SubFOp>(nestedLoc, args[0], args[1]);
          auto expOp =
              nestedBuilder.create<math::ExpOp>(nestedLoc, subOp.getResult());
          nestedBuilder.create<linalg::YieldOp>(loc, expOp->getResults());
        });

    // 3.a Build `linalg.generic` op for accumulator_init * exp(max_init -
    // max_result)
    SmallVector<AffineMap, 4> mapAccInit;
    // acc
    mapAccInit.push_back(AffineMap::getMultiDimIdentityMap(nloops - 1, ctx));
    // max
    mapAccInit.push_back(AffineMap::getMultiDimIdentityMap(nloops - 1, ctx));
    // max result
    mapAccInit.push_back(AffineMap::getMultiDimIdentityMap(nloops - 1, ctx));
    // init
    mapAccInit.push_back(AffineMap::getMultiDimIdentityMap(nloops - 1, ctx));
    SmallVector<Value> accInitInputs{op.accumulator(), op.max(), partialMax};

    auto reducedResultTy = partialMax.getType().cast<RankedTensorType>();
    auto reduceEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, reducedResultTy.getShape(), reducedResultTy.getElementType());

    auto linalgAccInitOp = rewriter.create<linalg::GenericOp>(
        loc, reducedResultTy, accInitInputs, reduceEmptyOp->getResults(),
        mapAccInit, getNParallelLoopsAttrs(nloops - 1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto subOp =
              nestedBuilder.create<arith::SubFOp>(nestedLoc, args[1], args[2]);
          auto expOp =
              nestedBuilder.create<math::ExpOp>(nestedLoc, subOp.getResult());
          auto mulOp = nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0],
                                                           expOp.getResult());
          nestedBuilder.create<linalg::YieldOp>(loc, mulOp->getResults());
        });

    // 3.b 3Build `linalg.reduce` op for acc
    auto partialAcc = createReduce<arith::AddFOp>(
        partialExpOp.getResult(0), linalgAccInitOp.getResult(0),
        op.getDimension(), loc, rewriter);

    // 4.a Build `linalg.generic` op for result, exp(x - max) / acc
    SmallVector<Value> divInputs{partialExpOp.getResult(0), partialAcc};
    auto partialResultOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, divInputs, emptyOp->getResults(), mapWithBroadcast,
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto divOp =
              nestedBuilder.create<arith::DivFOp>(nestedLoc, args[0], args[1]);
          nestedBuilder.create<linalg::YieldOp>(loc, divOp->getResults());
        });

    // 4.b Build `linalg.generic` op for scale, accumulator_init * exp(max_init
    // - max_result)/acc
    SmallVector<AffineMap, 4> mapScale;
    // result acc_reduce_init, aka accumulator_init * exp(max_init - max_result)
    mapScale.push_back(AffineMap::getMultiDimIdentityMap(nloops - 1, ctx));
    // acc
    mapScale.push_back(AffineMap::getMultiDimIdentityMap(nloops - 1, ctx));
    // init
    mapScale.push_back(AffineMap::getMultiDimIdentityMap(nloops - 1, ctx));
    SmallVector<Value> scaleInputs{linalgAccInitOp.getResult(0), partialAcc};

    auto linalgScaleOp = rewriter.create<linalg::GenericOp>(
        loc, reducedResultTy, scaleInputs, reduceEmptyOp->getResults(),
        mapScale, getNParallelLoopsAttrs(nloops - 1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto mulOp =
              nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0], args[1]);
          nestedBuilder.create<linalg::YieldOp>(loc, mulOp->getResults());
        });

    SmallVector<Value> results;
    results.push_back(partialResultOp.getResult(0));
    results.push_back(partialMax);
    results.push_back(partialAcc);
    results.push_back(linalgScaleOp.getResult(0));

    rewriter.replaceOp(op, results);

    partialResultOp->getParentOfType<func::FuncOp>().dump();
    return success();
  }
};

struct LinalgExtToLinalgPass
    : public LinalgExtToLinalgBase<LinalgExtToLinalgPass> {

  LinalgExtToLinalgPass() = default;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);

    target.addLegalDialect<tensor::TensorDialect, arith::ArithDialect,
                           linalg::LinalgDialect, math::MathDialect,
                           memref::MemRefDialect>();

    target.addIllegalOp<linalg_ext::SoftmaxOp>();
    populateLinalgExtToLinalgConversionPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateLinalgExtToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SoftmaxOpToLinalgReduceAndGeneric>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgExtToLinalgPass() {
  return std::make_unique<LinalgExtToLinalgPass>();
}
