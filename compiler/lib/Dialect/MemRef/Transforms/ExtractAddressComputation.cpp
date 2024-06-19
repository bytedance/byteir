//===- Transforms/ExtractAddressComputation.cpp ------------------- C++ -*-===//
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
// Some code comes from ExtractAddressComputation.cpp in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/MemRef/Transforms/ExtractAddressComputation.h"
#include "byteir/Dialect/MemRef/Utils/Layout.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {
/// Rewrite a store/load-like op so that all its indices are zeros.
/// E.g., %ld = memref.load %base[%off0]...[%offN]
/// =>
/// %new_base = subview %base[%off0,.., %offN][1,..,1][1,..,1]
/// %ld = memref.load %new_base[0,..,0] :
///    memref<1x..x1xTy, strided<[1,..,1], offset: ?>>
///
/// `getSrcMemRef` returns the source memref for the given load-like operation.
///
/// Using the given rewriter, `rebuildOpFromAddressAndIndices` creates a new
/// StoreLoadLikeOp that reads from srcMemRef[indices].
/// The returned operation will be used to replace storeLoadOp.
template <typename StoreLoadLikeOp, Value (*getSrcMemRef)(StoreLoadLikeOp),
          StoreLoadLikeOp (*rebuildOpFromAddressAndIndices)(
              RewriterBase & /*rewriter*/, StoreLoadLikeOp /*storeLoadOp*/,
              Value /*srcMemRef*/, ArrayRef<Value> /*indices*/),
          SmallVector<OpFoldResult> (*getViewSizeForEachDim)(
              RewriterBase & /*rewriter*/, StoreLoadLikeOp /*storeLoadOp*/)>
struct StoreLoadLikeOpRewriter : public OpRewritePattern<StoreLoadLikeOp> {
  using OpRewritePattern<StoreLoadLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreLoadLikeOp storeLoadLikeOp,
                                PatternRewriter &rewriter) const override {
    Value srcMemRef = getSrcMemRef(storeLoadLikeOp);
    auto ldTy = cast<MemRefType>(srcMemRef.getType());
    unsigned storeLoadRank = ldTy.getRank();
    // Don't waste compile time if there is nothing to rewrite.
    if (storeLoadRank == 0)
      return failure();

    // If our load already has only zeros as indices there is nothing
    // to do.
    SmallVector<OpFoldResult> indices =
        getAsOpFoldResult(storeLoadLikeOp.getIndices());
    if (std::all_of(indices.begin(), indices.end(),
                    [](const OpFoldResult &opFold) {
                      return isConstantIntValue(opFold, 0);
                    })) {
      return failure();
    }

    // Create the array of ones of the right size.
    SmallVector<OpFoldResult> ones(storeLoadRank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        getViewSizeForEachDim(rewriter, storeLoadLikeOp);
    assert(sizes.size() == storeLoadRank &&
           "Expected one size per load dimension");
    Location loc = storeLoadLikeOp.getLoc();
    auto subview =
        rewriter.create<memref::SubViewOp>(loc, /*source=*/srcMemRef,
                                           /*offsets=*/indices,
                                           /*sizes=*/sizes, /*strides=*/ones);
    // Rewrite the load with the subview as the base pointer.
    SmallVector<Value> zeros(storeLoadRank,
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
    StoreLoadLikeOp newLoad = rebuildOpFromAddressAndIndices(
        rewriter, storeLoadLikeOp, subview.getResult(), zeros);
    rewriter.replaceOp(storeLoadLikeOp, newLoad->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper functions for the `load base[off0...]`
//  => `load (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getSrcMemRef specs for LoadOp.
// \see LoadLikeOpRewriter.
static Value getLoadOpSrcMemRef(memref::LoadOp loadOp) {
  return loadOp.getMemRef();
}

// Matches rebuildOpFromAddressAndIndices specs for LoadOp.
// \see LoadLikeOpRewriter.
static memref::LoadOp rebuildLoadOp(RewriterBase &rewriter,
                                    memref::LoadOp loadOp, Value srcMemRef,
                                    ArrayRef<Value> indices) {
  Location loc = loadOp.getLoc();
  return rewriter.create<memref::LoadOp>(loc, srcMemRef, indices,
                                         loadOp.getNontemporal());
}

SmallVector<OpFoldResult> getLoadOpViewSizeForEachDim(RewriterBase &rewriter,
                                                      memref::LoadOp loadOp) {
  MemRefType ldTy = loadOp.getMemRefType();
  unsigned loadRank = ldTy.getRank();
  return SmallVector<OpFoldResult>(loadRank, rewriter.getIndexAttr(1));
}

//===----------------------------------------------------------------------===//
// Helper functions for the `store val, base[off0...]`
//  => `store val, (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getSrcMemRef specs for StoreOp.
// \see LoadStoreLikeOpRewriter.
static Value getStoreOpSrcMemRef(memref::StoreOp storeOp) {
  return storeOp.getMemRef();
}

// Matches rebuildOpFromAddressAndIndices specs for StoreOp.
// \see LoadStoreLikeOpRewriter.
static memref::StoreOp rebuildStoreOp(RewriterBase &rewriter,
                                      memref::StoreOp storeOp, Value srcMemRef,
                                      ArrayRef<Value> indices) {
  Location loc = storeOp.getLoc();
  return rewriter.create<memref::StoreOp>(loc, storeOp.getValueToStore(),
                                          srcMemRef, indices,
                                          storeOp.getNontemporal());
}

SmallVector<OpFoldResult>
getStoreOpViewSizeForEachDim(RewriterBase &rewriter, memref::StoreOp storeOp) {
  MemRefType ldTy = storeOp.getMemRefType();
  unsigned loadRank = ldTy.getRank();
  return SmallVector<OpFoldResult>(loadRank, rewriter.getIndexAttr(1));
}

void populateExtractAddressComputationPatterns(RewritePatternSet &patterns) {
  patterns.add<StoreLoadLikeOpRewriter<
                   memref::LoadOp,
                   /*getSrcMemRef=*/getLoadOpSrcMemRef,
                   /*rebuildOpFromAddressAndIndices=*/rebuildLoadOp,
                   /*getViewSizeForEachDim=*/getLoadOpViewSizeForEachDim>,
               StoreLoadLikeOpRewriter<
                   memref::StoreOp,
                   /*getSrcMemRef=*/getStoreOpSrcMemRef,
                   /*rebuildOpFromAddressAndIndices=*/rebuildStoreOp,
                   /*getViewSizeForEachDim=*/getStoreOpViewSizeForEachDim>>(
      patterns.getContext());
}

struct ExtractAddressComputationPass
    : public ExtractAddressComputationBase<ExtractAddressComputationPass> {
  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    populateExtractAddressComputationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createExtractAddressComputationPass() {
  return std::make_unique<ExtractAddressComputationPass>();
}