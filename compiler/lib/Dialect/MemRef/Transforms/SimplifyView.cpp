//===- SimplifyView.cpp ---------------------------------------*--- C++ -*-===//
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
// Some code comes from ComposeSubView.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns for combining composed subview ops (i.e. subview
// of a subview becomes a single subview).
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/MemRef/Transforms/SimplifyView.h"
#include "PassDetail.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;

namespace {

// Util linearize
std::optional<int64_t> getLinearizeOffset(ArrayRef<int64_t> offsets,
                                          ArrayRef<int64_t> sizes) {
  if (sizes.size() == 0)
    return 0;

  int64_t sum = 0;
  int64_t prod = 1;
  int rank = offsets.size();
  bool prevDynamic = false;
  for (int i = rank - 1; i >= 0; --i) {
    if (prevDynamic || offsets[i] == ShapedType::kDynamic) {
      return std::nullopt;
    }

    sum += offsets[i] * prod;

    if (sizes[i] == ShapedType::kDynamic) {
      prevDynamic = true;
    }
    prod *= sizes[i];
  }

  return sum;
}

inline int64_t getBytes(int64_t bits) { return (bits + 7) >> 3; }

struct ComposeSubViewOfView : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {

    auto srcView = op.getSource().getDefiningOp<memref::ViewOp>();
    if (!srcView)
      return failure();

    // A 'SubViewOp' can be "rank-reducing" by eliminating dimensions of the
    // output memref that are statically known to be equal to 1. We do not
    // allow 'sourceOp' to be a rank-reducing subview because then our two
    // 'SubViewOp's would have different numbers of offset/size/stride
    // parameters (just difficult to deal with, not impossible if we end up
    // needing it).
    if (srcView.getType().getRank() != op.getType().getRank()) {
      return failure();
    }

    // only support ContiguousRowMajor
    if (!isStaticShapeAndContiguousRowMajorEx(op.getType())) {
      return failure();
    }

    SmallVector<OpFoldResult> strides = op.getMixedStrides();

    // Because we only support input strides of 1, the output stride is also
    // always 1.
    if (llvm::all_of(strides, [](OpFoldResult &valueOrAttr) {
          Attribute attr = valueOrAttr.dyn_cast<Attribute>();
          return attr && attr.cast<IntegerAttr>().getInt() == 1;
        })) {
    } else {
      return failure();
    }

    // FIXME: only support constant for now
    SmallVector<int64_t> offsets;
    for (auto opOffset : op.getMixedOffsets()) {
      // We only support static offset.
      if (opOffset.is<Value>()) {
        return failure();
      }
      Attribute opOffsetAttr = opOffset.dyn_cast<Attribute>();
      offsets.push_back(opOffsetAttr.cast<IntegerAttr>().getInt());
    }

    auto maybeInt = getLinearizeOffset(offsets, srcView.getType().getShape());
    if (!maybeInt.has_value())
      return failure();

    int64_t offsetInByte =
        *maybeInt * getBytes(op.getType().getElementTypeBitWidth());

    auto lieanrizedOffset =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), offsetInByte);

    auto newShift = rewriter.create<arith::AddIOp>(
        op.getLoc(), srcView.getByteShift(), lieanrizedOffset.getResult());

    // New MemRefType
    auto newMemRefType = MemRefType::get(
        op.getType().getShape(), op.getType().getElementType(),
        MemRefLayoutAttrInterface{}, op.getType().getMemorySpace());

    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        op, newMemRefType, srcView.getSource(), newShift.getResult(),
        op.sizes());

    return success();
  }
};

struct ComposeViewOfView : public OpRewritePattern<memref::ViewOp> {
  using OpRewritePattern<memref::ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ViewOp op,
                                PatternRewriter &rewriter) const override {
    auto srcView = op.getSource().getDefiningOp<memref::ViewOp>();
    if (!srcView)
      return failure();

    auto newShift = rewriter.create<arith::AddIOp>(
        op.getLoc(), op.getByteShift(), srcView.getByteShift());

    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        op, op.getType(), srcView.getSource(), newShift.getResult(),
        op.getSizes());

    return success();
  }
};

// This is an enhanced version of upstream `ComposeSubViewOpPattern` Pattern.
// Replaces a subview of a subview with a single subview. Only supports subview
// ops with static sizes and static strides of 1 (both static and dynamic
// offsets are supported).
// TODO: submit a PR to LLVM
struct ComposeSubViewOfSubView : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    // 'op' is the 'SubViewOp' we're rewriting. 'sourceOp' is the op that
    // produces the input of the op we're rewriting (for 'SubViewOp' the input
    // is called the "source" value). We can only combine them if both 'op' and
    // 'sourceOp' are 'SubViewOp'.
    auto sourceOp = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!sourceOp)
      return failure();

    // A 'SubViewOp' can be "rank-reducing" by eliminating dimensions of the
    // output memref that are statically known to be equal to 1. We do not
    // allow 'sourceOp' to be a rank-reducing subview because then our two
    // 'SubViewOp's would have different numbers of offset/size/stride
    // parameters (just difficult to deal with, not impossible if we end up
    // needing it).
    if (sourceOp.getSourceType().getRank() != sourceOp.getType().getRank()) {
      return failure();
    }

    // Offsets, sizes and strides OpFoldResult for the combined 'SubViewOp'.
    SmallVector<OpFoldResult> offsets, sizes, strides;

    // The rules for calculating the new offsets, strides and sizes are:
    // * Multiple subview offsets for a given dimension compose by weighted
    // addition. (Composed SubViewOffset_i = curSubViewOffset_i *
    // sourceSubViewStride_i + sourceSubViewOffset_i)
    // * Multiple subview strides for a given dimension compose by multiplying.
    // (Composed SubViewStride_i = curSubViewStride_i * sourceSubViewStride_i)
    // * Multiple sizes for a given dimension compose by taking the size of the
    //   final subview and ignoring the rest. ("Take m values" followed by "Take
    //   n values" == "Take n values") This size must also be the smallest one
    //   by definition (a subview needs to be the same size as or smaller than
    //   its source along each dimension; presumably subviews that are larger
    //   than their sources are disallowed by validation).
    for (auto it : llvm::zip(op.getMixedOffsets(), sourceOp.getMixedOffsets(),
                             op.getMixedSizes(), sourceOp.getMixedStrides(),
                             op.getMixedStrides())) {
      auto opOffset = std::get<0>(it);
      auto sourceOffset = std::get<1>(it);
      auto opSize = std::get<2>(it);
      auto sourceStride = std::get<3>(it);
      auto opStride = std::get<4>(it);

      // We only support static sizes and strides.
      if (opSize.is<Value>() || sourceStride.is<Value>() ||
          opStride.is<Value>()) {
        return failure();
      }
      auto sourceStrideInt =
          sourceStride.dyn_cast<Attribute>().cast<IntegerAttr>().getInt();
      auto opStrideInt =
          opStride.dyn_cast<Attribute>().cast<IntegerAttr>().getInt();

      sizes.push_back(opSize);
      strides.push_back(
          rewriter.getI64IntegerAttr(sourceStrideInt * opStrideInt));
      Attribute opOffsetAttr = opOffset.dyn_cast<Attribute>(),
                sourceOffsetAttr = sourceOffset.dyn_cast<Attribute>();

      if (opOffsetAttr && sourceOffsetAttr) {
        // If both offsets are static we can simply calculate the combined
        // offset statically.
        offsets.push_back(rewriter.getI64IntegerAttr(
            opOffsetAttr.cast<IntegerAttr>().getInt() * sourceStrideInt +
            sourceOffsetAttr.cast<IntegerAttr>().getInt()));
      } else {
        // When either offset is dynamic, we must emit an additional affine
        // transformation to add the two offsets together dynamically.
        SmallVector<Value> affineApplyOperands;
        auto getExpr = [&affineApplyOperands,
                        &rewriter](OpFoldResult attrOrValue) {
          if (auto attr = attrOrValue.dyn_cast<Attribute>()) {
            return rewriter.getAffineConstantExpr(
                attr.cast<IntegerAttr>().getInt());
          } else {
            auto ret = rewriter.getAffineSymbolExpr(affineApplyOperands.size());
            affineApplyOperands.push_back(attrOrValue.get<Value>());
            return ret;
          }
        };
        AffineExpr expr =
            getExpr(opOffset) * sourceStrideInt + getExpr(sourceOffset);

        AffineMap map = AffineMap::get(0, affineApplyOperands.size(), expr);
        Value result = rewriter.create<AffineApplyOp>(op.getLoc(), map,
                                                      affineApplyOperands);
        offsets.push_back(result);
      }
    }

    // This replaces 'op' but leaves 'sourceOp' alone; if it no longer has any
    // uses it can be removed by a (separate) dead code elimination pass.
    rewriter.replaceOpWithNewOp<memref::SubViewOp>(op, sourceOp.getSource(),
                                                   offsets, sizes, strides);
    return success();
  }
};

struct SimplifyViewPass : public SimplifyViewBase<SimplifyViewPass> {
public:
  SimplifyViewPass() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateSimplifyViewPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("SimplifyViewPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateSimplifyViewPattern(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ComposeViewOfView, 
               ComposeSubViewOfView, 
               ComposeSubViewOfSubView>(patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createSimplifyViewPass() {
  return std::make_unique<SimplifyViewPass>();
}
