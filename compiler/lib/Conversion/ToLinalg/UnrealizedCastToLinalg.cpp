//===- UnrealizedCastToLinalg.cpp -----------------------------*--- C++ -*-===//
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
// Some code comes from legalize_to_linalg.cc in TensorFlow
// Original license:
//
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
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
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToLinalg/ToLinalg.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;

namespace {

static SmallVector<Value, 2> extractDynamicSizes(OpBuilder &b, Location loc,
                                                 Value tensor,
                                                 Value shape_tensor = nullptr,
                                                 AffineMap permutation = {}) {
  auto tensor_type = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensor_type)
    return {};
  SmallVector<Value, 2> dynSizes(tensor_type.getRank());
  for (const auto &en : llvm::enumerate(tensor_type.getShape())) {
    if (en.value() != ShapedType::kDynamic)
      continue;
    // If a shape tensor is present extract from there.
    if (shape_tensor) {
      Value extract = b.create<tensor::ExtractOp>(
          loc, shape_tensor,
          ValueRange{b.create<ConstantIndexOp>(loc, en.index())});
      dynSizes[en.index()] =
          b.create<IndexCastOp>(loc, b.getIndexType(), extract);
    } else {
      dynSizes[en.index()] = b.create<tensor::DimOp>(loc, tensor, en.index());
    }
  }
  if (permutation)
    dynSizes = applyPermutationMap(permutation, ArrayRef<Value>(dynSizes));
  llvm::erase_value(dynSizes, nullptr); // Strip out placeholders.
  return dynSizes;
}

static Value getInitTensor(OpBuilder &b, Location loc, ShapedType type,
                           ArrayRef<Value> dyn_sizes) {
  return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(),
                                   dyn_sizes);
}

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

class UnrealizedCastToLinalgConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op,
                  UnrealizedConversionCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Find maximum rank / number of loops.
    auto getRank = [](Value v) {
      return cast<ShapedType>(v.getType()).getRank();
    };

    auto isScalar = [&](Value v) { return getRank(v) == 0; };
    Value maxRankArg = adaptor.getOperands().front();
    int64_t nloops = getRank(maxRankArg);

    // Find result type, if on tensors.
    ShapedType resultTy = dyn_cast<ShapedType>(op->getResultTypes().front());

    // Find input/output values and types.
    auto loc = op.getLoc();
    ValueRange inputs = adaptor.getOperands();

    auto dynSizes = extractDynamicSizes(rewriter, loc, maxRankArg);

    Value output = getInitTensor(rewriter, loc, resultTy, dynSizes);

    // Create indexing maps.
    AffineMap scalarMap = AffineMap::get(nloops, 0, rewriter.getContext());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(nloops);
    SmallVector<AffineMap, 4> maps;
    for (Value v : adaptor.getOperands()) {
      maps.push_back(isScalar(v) ? scalarMap : idMap);
    }
    maps.push_back(idMap);

    // Build `linalg.generic` op.
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, inputs, output, maps, getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Type innerResultTy = getElementTypeOrSelf(output);

          auto innerCastOp = nestedBuilder.create<UnrealizedConversionCastOp>(
              loc, innerResultTy,
              llvm::to_vector<2>(args.take_front(inputs.size())));
          Value innerResult = innerCastOp.getResult(0);

          nestedBuilder.create<linalg::YieldOp>(loc, innerResult);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct UnrealizedCastToLinalgPass
    : public UnrealizedCastToLinalgBase<UnrealizedCastToLinalgPass> {

  UnrealizedCastToLinalgPass() = default;

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
                           math::MathDialect, tensor::TensorDialect,
                           scf::SCFDialect, shape::ShapeDialect>();

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          return !(isa<TensorType>(op.getOperand(0).getType()) &&
                   isa<TensorType>(op.getResult(0).getType()));
        });

    populateUnrealizedCastToLinalgConversionPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateUnrealizedCastToLinalgConversionPattern(
    RewritePatternSet &patterns) {
  patterns.add<UnrealizedCastToLinalgConverter>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createUnrealizedCastToLinalgPass() {
  return std::make_unique<UnrealizedCastToLinalgPass>();
}
