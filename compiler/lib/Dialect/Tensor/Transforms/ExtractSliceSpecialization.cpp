//===- ExtractSliceSpecialization.cpp ----------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "byteir/Dialect/Tensor/Transforms/ExtractSliceSpecialization.h"

#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/TransformUtils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

#define DEBUG_TYPE "extract-pad-specialization"

using namespace mlir;

namespace {
static bool isNormalizeExtractSlice(tensor::ExtractSliceOp extractSliceOp) {
  ArrayRef<int64_t> offsets = extractSliceOp.getStaticOffsets();
  ArrayRef<int64_t> strides = extractSliceOp.getStaticStrides();
  if (!llvm::all_of(offsets, [](int64_t v) { return v == 0; })) {
    return false;
  }

  if (!llvm::all_of(strides, [](int64_t v) { return v == 1; })) {
    return false;
  }
  return true;
}

static std::optional<OpFoldResult>
reifyDimsInSliceOp(Operation *op, RewriterBase &rewriter, int64_t dim) {
  if (auto expandOp = llvm::dyn_cast<tensor::ExpandShapeOp>(op)) {
    auto ReassociationIndices = expandOp.getReassociationIndices();
    auto src = expandOp.getSrc();
    RankedTensorType resType = expandOp.getType();
    if (RankedTensorType srcType = src.getType().dyn_cast<RankedTensorType>()) {
      int64_t dynDimCount = 0;

      if (resType.getShape()[dim] != ShapedType::kDynamic) {
        return std::nullopt;
      }
      int srcDim = expandOp.getCorrespondingSourceDim(dim);
      for (auto idx : ReassociationIndices[srcDim]) {

        if (resType.getShape()[idx] == ShapedType::kDynamic) {
          dynDimCount += 1;
        } else if (resType.getShape()[idx] != 1) {
          return std::nullopt;
        }
      }
      if (src.getDefiningOp() && dynDimCount == 1) {
        return reifyDimsInSliceOp(src.getDefiningOp(), rewriter, srcDim);
      }
    }
  } else if (auto extractSliceOp = llvm::dyn_cast<tensor::ExtractSliceOp>(op)) {
    ReifiedRankedShapedTypeDims reifiedShapes;
    if (failed(extractSliceOp.reifyResultShapes(rewriter, reifiedShapes))) {
      return std::nullopt;
    }
    return reifiedShapes[0][dim];
  }
  return std::nullopt;
}

/// When the shape of extracted_slice equal to input tensor,
/// convert fill + extracted_slice to fill + collaspe_slice.
///
/// Example:
///  %0 = tensor.extract_slice %arg2[0, %arg1] [%dim_0, 1] [1, 1] :
///  tensor<?x32xf32> to tensor<?xf32> %1 = tensor.expand_shape %0 [[0, 1]] :
///  tensor<?xf32> into tensor<?x1xf32> %2 = linalg.fill ins(%cst : f32) outs(%1
///  : tensor<?x1xf32>) -> tensor<?x1xf32> %3 = tensor.extract_slice %2[0, 0]
///  [%dim_0, 1] [1, 1] : tensor<?x1xf32> to tensor<?xf32>
///
/// will be converted to
///
///  %0 = tensor.extract_slice %arg2[0, %arg1] [%dim_0, 1] [1, 1] :
///  tensor<?x32xf32> to tensor<?xf32> %1 = tensor.expand_shape %0 [[0, 1]] :
///  tensor<?xf32> into tensor<?x1xf32> %2 = linalg.fill ins(%cst : f32) outs(%1
///  : tensor<?x1xf32>) -> tensor<?x1xf32> %3 = tensor.collapse_shape %2[0, 0]
///  [%dim_0, 1] [1, 1] : tensor<?x1xf32> to tensor<?xf32>
struct ExtractFullSliceFromLinalgFillOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {

  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!isNormalizeExtractSlice(extractSliceOp)) {
      return failure();
    }

    linalg::FillOp fillOp =
        extractSliceOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp || fillOp.getNumResults() != 1) {
      return failure();
    }

    ReifiedRankedShapedTypeDims fillReifiedShapes;
    if (failed(fillOp.reifyResultShapes(rewriter, fillReifiedShapes))) {
      return failure();
    }

    // reifyDims if getDefiningOp is expandShape/extractSlice.
    for (size_t i = 0; i < fillReifiedShapes[0].size(); ++i) {
      auto maybeCst = getConstantIntValue(fillReifiedShapes[0][i]);
      if (maybeCst.has_value())
        continue;
      Value val = fillReifiedShapes[0][i].get<Value>();
      if (auto dimOp = val.getDefiningOp<tensor::DimOp>()) {
        auto maybeCstIdx = dimOp.getConstantIndex();
        if (maybeCstIdx.has_value() && dimOp.getSource().getDefiningOp()) {
          auto reifedDim = reifyDimsInSliceOp(dimOp.getSource().getDefiningOp(),
                                              rewriter, maybeCstIdx.value());
          if (reifedDim.has_value())
            fillReifiedShapes[0][i] = reifedDim.value();
        }
      }
    }

    SmallVector<OpFoldResult> sizes = extractSliceOp.getMixedSizes();
    if (!isEqualConstantIntOrValueArray(sizes, fillReifiedShapes[0])) {
      return failure();
    }

    RankedTensorType resultType = extractSliceOp.getType();
    if (resultType.getRank() ==
        static_cast<int64_t>(fillReifiedShapes[0].size())) {
      rewriter.replaceOp(extractSliceOp, fillOp.getResult(0));
      return success();
    }

    // convert extract_slice to collapse_shape
    SmallVector<ReassociationIndices> reassociation;
    int64_t srcIdx = 0;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (resultType.getShape()[i] == 1) {
        reassociation.emplace_back(ReassociationIndices{srcIdx});
        srcIdx += 1;
      } else {
        ReassociationIndices indices;
        while (srcIdx < static_cast<int64_t>(sizes.size())) {
          indices.emplace_back(srcIdx);
          auto maybeCst = getConstantIntValue(sizes[srcIdx]);
          srcIdx += 1;
          if (!maybeCst.has_value() || maybeCst.value() != 1) {
            break;
          }
        }
        reassociation.emplace_back(indices);
      }
    }

    // reassociation is empty if tensor with rank 0
    if (resultType.getRank() > 0) {
      while (srcIdx < static_cast<int64_t>(sizes.size())) {
        reassociation.back().emplace_back(srcIdx);
        srcIdx += 1;
      }
    }

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        extractSliceOp, fillOp.getResult(0), reassociation);

    return success();
  }
};

struct ExtractFullSliceFromSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp curSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!isNormalizeExtractSlice(curSliceOp)) {
      return failure();
    }

    tensor::ExtractSliceOp preSliceOp =
        curSliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!preSliceOp)
      return failure();
    RankedTensorType preResultType = preSliceOp.getType();
    RankedTensorType curResultType = curSliceOp.getType();
    if (preResultType.getRank() != curResultType.getRank()) {
      return failure();
    }

    if (!isNormalizeExtractSlice(preSliceOp)) {
      return failure();
    }

    SmallVector<OpFoldResult> preSizes = preSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> curSizes = curSliceOp.getMixedSizes();
    preSizes.erase(std::remove_if(preSizes.begin(), preSizes.end(),
                                  [](OpFoldResult ofr) {
                                    auto maybeCst = getConstantIntValue(ofr);
                                    return maybeCst.has_value() &&
                                           maybeCst.value() == 1;
                                  }),
                   preSizes.end());

    curSizes.erase(std::remove_if(curSizes.begin(), curSizes.end(),
                                  [](OpFoldResult ofr) {
                                    auto maybeCst = getConstantIntValue(ofr);
                                    return maybeCst.has_value() &&
                                           maybeCst.value() == 1;
                                  }),
                   curSizes.end());
    if (!isEqualConstantIntOrValueArray(preSizes, curSizes)) {
      return failure();
    }

    rewriter.replaceOp(curSliceOp, preSliceOp.getResult());
    return success();
  }
};

struct ExtractSliceSpecializationPass
    : public ExtractSliceSpecializationBase<ExtractSliceSpecializationPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ExtractFullSliceFromLinalgFillOp, ExtractFullSliceFromSlice>(
        ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createExtractSliceSpecializationPass() {
  return std::make_unique<ExtractSliceSpecializationPass>();
}
