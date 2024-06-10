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
#include "byteir/Utils/AttrUtils.h"
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
#include "llvm/ADT/SmallSet.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;

namespace {

// utils

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
static SmallVector<utils::IteratorType>
getParallelAndReductionIterators(unsigned nLoops, unsigned nReduction) {
  SmallVector<utils::IteratorType> res(nLoops - nReduction,
                                       utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);

  return res;
}

static SmallVector<utils::IteratorType>
getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

// create linalg.reduce with a set of reductionDims
template <typename T>
static Value createReduce(Value input, Value init,
                          ArrayRef<int64_t> reductionDims, Location loc,
                          OpBuilder &builder) {

  SmallVector<Value> inputs{input};
  SmallVector<Value> inits{init};
  auto reduceOp = builder.create<linalg::ReduceOp>(
      loc, inputs, inits, reductionDims,
      [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        auto innerOp = nestedBuilder.create<T>(nestedLoc, args);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, innerOp->getResults());
      });
  return reduceOp.getResult(0);
}

// create linalg.reduce with a single of reductionDim
template <typename T>
static Value createReduce(Value input, Value init, int64_t reductionDim,
                          Location loc, OpBuilder &builder) {
  SmallVector<int64_t> reductionDims{reductionDim};
  return createReduce<T>(input, init, reductionDims, loc, builder);
}

static Value fillTensorWithZeros(OpBuilder &builder, Location loc,
                                 Value tensor) {
  auto type = cast<ShapedType>(tensor.getType());
  Value zero;
  auto zeroAttr = builder.getZeroAttr(type.getElementType());
  zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
  return builder.create<linalg::FillOp>(loc, zero, tensor).result();
}

static SmallVector<int64_t> getShapeWithSkips(ArrayRef<int64_t> shape,
                                              ArrayRef<int64_t> skips) {
  SmallVector<int64_t> ret;
  llvm::SmallSet<int64_t, 4> skipSet;
  skipSet.insert(skips.begin(), skips.end());
  int64_t numDims = shape.size();
  for (int64_t i = 0; i < numDims; ++i) {
    if (skipSet.contains(i)) {
      continue;
    }
    ret.push_back(shape[i]);
  }

  return ret;
}

// TODO support dynamic
static int64_t reduceStaticDivisor(ArrayRef<int64_t> shape,
                                   ArrayRef<int64_t> dims) {
  int64_t prod = 1;
  for (auto d : dims) {
    prod *= shape[d];
  }
  return prod;
}

struct LayerNormOpToLinalgReduceAndGeneric
    : public OpConversionPattern<linalg_ext::LayerNormOp> {
  using OpConversionPattern<linalg_ext::LayerNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg_ext::LayerNormOp op,
                  linalg_ext::LayerNormOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // check op has tensor semantics first
    if (!op.hasTensorSemantics()) {
      return failure();
    }

    auto ctx = op.getContext();
    auto loc = op.getLoc();

    // 1. Build a sum
    // using weight Ty as reduceTy since they are same
    auto inputTy = cast<RankedTensorType>(op.getOperandType(0));
    auto reduceShape = getShapeWithSkips(inputTy.getShape(), op.getIntAxis());
    auto reduceTy =
        RankedTensorType::get(reduceShape, inputTy.getElementType());

    // Create an empty
    auto reduceEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, reduceShape, inputTy.getElementType());
    auto zeroVal =
        fillTensorWithZeros(rewriter, loc, reduceEmptyOp.getResult());
    auto sumVal = createReduce<arith::AddFOp>(op.input(), zeroVal,
                                              op.getIntAxis(), loc, rewriter);

    // 2. Build a mean
    int64_t reduceloops = reduceShape.size();
    // calculate maps for mean
    SmallVector<AffineMap, 4> meanMap;
    // input (sum)
    meanMap.push_back(AffineMap::getMultiDimIdentityMap(reduceloops, ctx));
    // init
    meanMap.push_back(AffineMap::getMultiDimIdentityMap(reduceloops, ctx));
    SmallVector<Value> meanInputs{sumVal};

    auto reduceDivisor =
        reduceStaticDivisor(inputTy.getShape(), op.getIntAxis());
    auto meanOp = rewriter.create<linalg::GenericOp>(
        loc, reduceTy, meanInputs, reduceEmptyOp->getResults(), meanMap,
        getNParallelLoopsAttrs(reduceloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto conOp = nestedBuilder.create<arith::ConstantOp>(
              nestedLoc, nestedBuilder.getFloatAttr(reduceTy.getElementType(),
                                                    reduceDivisor));
          auto divOp = nestedBuilder.create<arith::DivFOp>(nestedLoc, args[0],
                                                           conOp.getResult());
          nestedBuilder.create<linalg::YieldOp>(loc, divOp->getResults());
        });

    // 3. Build sum(X^2)
    SmallVector<Value> SqureInputs{op.input()};
    SmallVector<Value> SqureInits{zeroVal};
    auto reduceSumSqureOp = rewriter.create<linalg::ReduceOp>(
        loc, SqureInputs, SqureInits, op.getIntAxis(),
        [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto squreOp =
              nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0], args[0]);
          auto innerAddOp = nestedBuilder.create<arith::AddFOp>(
              nestedLoc, squreOp.getResult(), args[1]);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc,
                                                innerAddOp->getResults());
        });

    // 4. Build Var
    // calculate maps for var
    SmallVector<AffineMap, 4> varMap;
    // input (sum(X^2) and mean)
    varMap.push_back(AffineMap::getMultiDimIdentityMap(reduceloops, ctx));
    varMap.push_back(AffineMap::getMultiDimIdentityMap(reduceloops, ctx));
    // init
    varMap.push_back(AffineMap::getMultiDimIdentityMap(reduceloops, ctx));
    SmallVector<Value> varInputs{reduceSumSqureOp.getResult(0),
                                 meanOp.getResult(0)};

    auto VarOp = rewriter.create<linalg::GenericOp>(
        loc, reduceTy, varInputs, reduceEmptyOp->getResults(), varMap,
        getNParallelLoopsAttrs(reduceloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // N
          auto constOp = nestedBuilder.create<arith::ConstantOp>(
              nestedLoc, nestedBuilder.getFloatAttr(reduceTy.getElementType(),
                                                    reduceDivisor));
          // sum(X^2)/N
          auto divOp = nestedBuilder.create<arith::DivFOp>(nestedLoc, args[0],
                                                           constOp.getResult());
          // mean^2
          auto squreMeanOp =
              nestedBuilder.create<arith::MulFOp>(nestedLoc, args[1], args[1]);
          // sum(X^2)/N - mean^2
          auto subOp = nestedBuilder.create<arith::SubFOp>(
              nestedLoc, divOp.getResult(), squreMeanOp.getResult());

          nestedBuilder.create<linalg::YieldOp>(loc, subOp->getResults());
        });

    // 5. Build Output
    // output = (input - mean) * rsqrt(var + eps) * weight + bias
    int64_t nloops = inputTy.getRank();
    // calculate maps for var
    SmallVector<AffineMap, 4> finalMap;
    // input (input, mean, var, weight, bias)
    finalMap.push_back(AffineMap::getMultiDimIdentityMap(nloops, ctx));
    finalMap.push_back(
        getMultiDimIdentityMapWithSkips(nloops, op.getIntAxis(), ctx));
    finalMap.push_back(
        getMultiDimIdentityMapWithSkips(nloops, op.getIntAxis(), ctx));
    finalMap.push_back(
        getMultiDimIdentityMapWithTargets(nloops, op.getIntAxis(), ctx));
    finalMap.push_back(
        getMultiDimIdentityMapWithTargets(nloops, op.getIntAxis(), ctx));
    // init
    finalMap.push_back(AffineMap::getMultiDimIdentityMap(nloops, ctx));
    SmallVector<Value> finalInputs{op.input(), meanOp.getResult(0),
                                   VarOp.getResult(0), op.weight(), op.bias()};
    // Create an empty
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, inputTy.getShape(),
                                                    inputTy.getElementType());
    // get epsAttr if cast needed
    auto epsAttr = castFloatAttr(op.getEpsilonAttr(), inputTy.getElementType());

    // compute output = (input - mean) * rsqrt(var + eps) * weight + bias
    // args = (input, mean, var, weight, bias)
    auto finalOp = rewriter.create<linalg::GenericOp>(
        loc, inputTy, finalInputs, emptyOp->getResults(), finalMap,
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // eps
          auto constOp =
              nestedBuilder.create<arith::ConstantOp>(nestedLoc, epsAttr);
          // var + eps
          auto addOp1 = nestedBuilder.create<arith::AddFOp>(
              nestedLoc, args[2], constOp.getResult());
          // rsqrt(var + eps)
          auto rsqrtOp = nestedBuilder.create<math::RsqrtOp>(
              nestedLoc, addOp1.getResult());
          // input - mean
          auto subOp =
              nestedBuilder.create<arith::SubFOp>(nestedLoc, args[0], args[1]);
          // (input - mean) * rsqrt(var + eps)
          auto mulOp0 = nestedBuilder.create<arith::MulFOp>(
              nestedLoc, rsqrtOp.getResult(), args[3]);
          // (input - mean) * rsqrt(var + eps) * weight
          auto mulOp1 = nestedBuilder.create<arith::MulFOp>(
              nestedLoc, subOp.getResult(), mulOp0.getResult());
          // (input - mean) * rsqrt(var + eps) * weight + bias
          auto addOp2 = nestedBuilder.create<arith::AddFOp>(
              nestedLoc, mulOp1.getResult(), args[4]);
          nestedBuilder.create<linalg::YieldOp>(loc, addOp2->getResults());
        });

    SmallVector<Value> results;
    results.push_back(finalOp.getResult(0));
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct SoftmaxOpToLinalgReduceAndGeneric
    : public OpConversionPattern<linalg_ext::SoftmaxOp> {
public:
  using OpConversionPattern<linalg_ext::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg_ext::SoftmaxOp op,
                  linalg_ext::SoftmaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Check op has tensor semantics here
    if (!op.hasTensorSemantics()) {
      return failure();
    }

    auto ctx = op.getContext();
    auto loc = op.getLoc();

    // 1. Build Max
    auto partialMax = createReduce<arith::MaxNumFOp>(
        op.input(), op.max(), op.getDimension(), loc, rewriter);

    // 2. Build exp(x-max)
    auto resultTy = cast<RankedTensorType>(op.getType(0));
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
    SmallVector<Value> expInputs{op.input(), partialMax};

    auto partialExpOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, /*inputs=*/expInputs, /*outputs=*/emptyOp->getResults(),
        mapWithBroadcast, getNParallelLoopsAttrs(nloops),
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

    auto reducedResultTy = cast<RankedTensorType>(partialMax.getType());
    auto reduceEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, reducedResultTy.getShape(), reducedResultTy.getElementType());

    auto linalgAccInitOp = rewriter.create<linalg::GenericOp>(
        loc, reducedResultTy, /*inputs=*/accInitInputs,
        /*outputs=*/reduceEmptyOp->getResults(), mapAccInit,
        getNParallelLoopsAttrs(nloops - 1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto subOp =
              nestedBuilder.create<arith::SubFOp>(nestedLoc, args[1], args[2]);
          auto expOp =
              nestedBuilder.create<math::ExpOp>(nestedLoc, subOp.getResult());
          auto mulOp = nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0],
                                                           expOp.getResult());
          nestedBuilder.create<linalg::YieldOp>(loc, mulOp->getResults());
        });

    // 3.b Build `linalg.reduce` op for acc
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

    SmallVector<Value, 4> results;
    results.push_back(partialResultOp.getResult(0));
    results.push_back(partialMax);
    results.push_back(partialAcc);
    results.push_back(linalgScaleOp.getResult(0));
    rewriter.replaceOp(op, results);
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
  //  patterns.add<SoftmaxOpToLinalgReduceAndGeneric>(patterns.getContext());
  patterns.add<SoftmaxOpToLinalgReduceAndGeneric,
               LayerNormOpToLinalgReduceAndGeneric>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgExtToLinalgPass() {
  return std::make_unique<LinalgExtToLinalgPass>();
}
