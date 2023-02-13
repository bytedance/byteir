//===- LmhloToMemref.cpp --------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/LmhloToMemref/LmhloToMemref.h"
#include "byteir/Utils/Utils.h"
#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::lmhlo;
using namespace mlir::memref;

namespace {

int64_t prod(ArrayRef<int64_t> a) {
  int64_t ret = 1;
  for (size_t i = 0; i < a.size(); ++i)
    ret *= a[i];
  return ret;
}

struct ConvertReshape : public OpRewritePattern<lmhlo::ReshapeOp> {
  using OpRewritePattern<lmhlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(lmhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // handles static shape only
    auto allocOp = op.getOutput().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();
    auto inMemRefType = op.getOperand().getType().cast<MemRefType>();
    auto outMemRefType = op.getOutput().getType().cast<MemRefType>();
    auto inputShape = inMemRefType.getShape();
    auto outputShape = outMemRefType.getShape();

    // check: product of output's shape must equal to operand's shape
    if (prod(inputShape) != prod(outputShape))
      return failure();

    // create meta memref of output shape
    SmallVector<int64_t> shape;
    shape.push_back(outputShape.size());
    auto shapeMetaMemRefType = MemRefType::get(shape, rewriter.getI64Type());
    auto shapeAllocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), shapeMetaMemRefType);
    auto constOp = rewriter.create<lmhlo::ConstantOp>(
        op.getLoc(),
        getI64ElementsAttr(outputShape, outputShape.size(), &rewriter),
        shapeAllocOp.getResult());

    auto newMemRefType = MemRefType::get(
        outMemRefType.getShape(), outMemRefType.getElementType(),
        outMemRefType.getLayout(), inMemRefType.getMemorySpace());
    auto newReshapeOp = rewriter.create<memref::ReshapeOp>(
        op.getLoc(), newMemRefType, op.getOperand(), constOp.getOutput());
    rewriter.replaceOp(allocOp, newReshapeOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct SliceToSubview : public OpRewritePattern<lmhlo::SliceOp> {
  using OpRewritePattern<lmhlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(lmhlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto allocOp = op.getOutput().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();
    auto inMemRefType = op.getOperand().getType().cast<MemRefType>();
    auto startIndices = SmallVector<int64_t>();
    auto limitIndices = SmallVector<int64_t>();
    auto strides = SmallVector<int64_t>();
    getValuesFromDenseIntElementsAttr(op.getStartIndices(), startIndices);
    getValuesFromDenseIntElementsAttr(op.getLimitIndices(), limitIndices);
    getValuesFromDenseIntElementsAttr(op.getStrides(), strides);
    auto inputShape = inMemRefType.getShape();

    if (startIndices.size() != limitIndices.size() ||
        limitIndices.size() != strides.size())
      return failure();
    // check: 0 <= start_indices[d] < limit_indices[d] < full_dim[d]
    // check: (limit_indices[d] - start_indices[d]) % strides[d] == 0
    for (size_t i = 0; i < startIndices.size(); ++i) {
      if (!(0 <= startIndices[i] && startIndices[i] < limitIndices[i] &&
            limitIndices[i] <= inputShape[i]))
        return failure();
      if ((limitIndices[i] - startIndices[i]) % strides[i] > 0)
        return failure();
    }

    SmallVector<int64_t> sizes;
    for (size_t i = 0; i < startIndices.size(); ++i)
      sizes.push_back((limitIndices[i] - startIndices[i]) / strides[i]);

    auto newSubViewOp = rewriter.create<memref::SubViewOp>(
        op.getLoc(), op.getOperand(), ArrayRef<int64_t>(startIndices),
        ArrayRef<int64_t>(sizes), ArrayRef<int64_t>(strides));
    rewriter.replaceOp(allocOp, newSubViewOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct LmhloToMemrefPass : public LmhloToMemrefBase<LmhloToMemrefPass> {
public:
  LmhloToMemrefPass() = default;
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateLmhloToMemrefPattern(patterns);
    target.addIllegalOp<lmhlo::ReshapeOp, lmhlo::SliceOp>();

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("LmhloToMemrefPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateLmhloToMemrefPattern(RewritePatternSet &patterns) {
  patterns.add<ConvertReshape, SliceToSubview>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLmhloToMemrefPass() {
  return std::make_unique<LmhloToMemrefPass>();
}
