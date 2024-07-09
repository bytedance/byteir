//===- DecomposeOnTorch.cpp -----------------------------------*--- C++ -*-===//
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
// Some code comes from Torch-MLIR in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-frontend/Dialect/Torch/Transforms/DecomposeOnTorch.h"
#include "./PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

// Helper function to compute the return type of the reduction function.
// `dim` specifies the dimension to reduce and `keepDim` preserves the rank of
// the input tensor.
static Type computeReductionType(PatternRewriter &rewriter, Operation *op,
                                 BaseTensorType tensorType, Value dim,
                                 bool keepDim) {
  SmallVector<int64_t> sizes;
  int64_t dimInt;
  if (tensorType.hasSizes()) {
    ArrayRef<int64_t> inputShape = tensorType.getSizes();
    int64_t inputRank = inputShape.size();
    if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
      dimInt = toPositiveDim(dimInt, inputRank);
      if (!isValidDim(dimInt, inputRank)) {
        (void)rewriter.notifyMatchFailure(op, "dim is not a valid dim");
        return nullptr;
      }
      sizes.append(inputShape.begin(), inputShape.end());
      // The dimension to be reduced is set to 1 when `keepDim` is true else it
      // is removed.
      if (keepDim)
        sizes[dimInt] = 1;
      else
        sizes.erase(sizes.begin() + dimInt);
    } else {
      unsigned reducedRank = keepDim ? inputRank : inputRank - 1;
      sizes.resize(reducedRank, kUnknownSize);
    }
  }

  Type resultType = tensorType.getWithSizesAndDtypeAndSparsity(
      !tensorType.hasSizes() ? std::optional<ArrayRef<int64_t>>()
                             : llvm::ArrayRef(sizes),
      tensorType.getOptionalDtype(), tensorType.getOptionalSparsity());
  return resultType;
}

template <typename OpTy>
static LogicalResult calculateVariance(OpTy op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value self = op.getSelf();
  Value dimList = op.getDim();
  Value keepDim = op.getKeepdim();
  BaseTensorType inputTensorTy = cast<BaseTensorType>(self.getType());
  Type outputType = op.getType();
  BaseTensorType outputTensorType = cast<BaseTensorType>(outputType);
  if (!outputTensorType.hasDtype()) {
    return rewriter.notifyMatchFailure(op,
                                       "expected result type to have a dtype");
  }
  if (!inputTensorTy.hasDtype() ||
      !isa<mlir::FloatType>(inputTensorTy.getDtype())) {
    return rewriter.notifyMatchFailure(
        op, "support floating-point type input only");
  }

  std::optional<unsigned> maybeInputRank = getTensorRank(self);
  if (!maybeInputRank) {
    return rewriter.notifyMatchFailure(op, "expected input to have a rank");
  }
  unsigned inputRank = *maybeInputRank;
  SmallVector<Value> dimListElements;
  bool isNoneOrEmpty = true;
  if (!isa<Torch::NoneType>(dimList.getType())) {
    if (!getListConstructElements(dimList, dimListElements))
      return rewriter.notifyMatchFailure(
          op, "expect dimList to be constructed from list construct");
    if (!dimListElements.empty() || inputRank == 0)
      isNoneOrEmpty = false;
  }
  if (isNoneOrEmpty) {
    for (unsigned i = 0; i < inputRank; i++)
      dimListElements.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        dimListElements);
  }
  Type meanDimResultType = inputTensorTy;
  for (unsigned i = 0; i < dimListElements.size(); i++)
    meanDimResultType = computeReductionType(
        rewriter, op, cast<BaseTensorType>(meanDimResultType),
        dimListElements[i],
        /*keepDim=*/true);

  Value constantNone = rewriter.create<ConstantNoneOp>(loc);
  Value constantTrue = rewriter.create<ConstantBoolOp>(loc, true);
  Value constantOne =
      rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
  Value meanAlongDims = rewriter.create<AtenMeanDimOp>(
      loc, meanDimResultType, self, dimList, /*keepDim=*/constantTrue,
      /*dtype=*/constantNone);
  Value subMean = rewriter.create<AtenSubTensorOp>(loc, inputTensorTy, self,
                                                   meanAlongDims, constantOne);
  Value square = rewriter.create<AtenSquareOp>(loc, inputTensorTy, subMean);

  Value result = rewriter.create<AtenMeanDimOp>(
      loc, outputTensorType, square, dimList, keepDim, /*dtype=*/constantNone);
  rewriter.replaceOp(op, result);
  return success();
}

// Decompose aten.var(x, dims) into:
// sub = aten.sub(x, aten.mean(x, dims))
// square = aten.square(sub)
// For Unbiased case:
// out = aten.sum(square, dims) / (productDimSize-1)
// For Biased case:
// out = aten.mean(square, dims)
struct DecomposeAtenVarDimOp : public OpRewritePattern<AtenVarDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarDimOp op,
                                PatternRewriter &rewriter) const override {
    bool unbiased;
    if (!matchPattern(op.getUnbiased(), m_TorchConstantBool(&unbiased))) {
      return rewriter.notifyMatchFailure(
          op, "Only support constant unbiased for aten.var");
    }
    if (unbiased) {
      return rewriter.notifyMatchFailure(op, "Only support biased variance");
    }
    if (failed(calculateVariance<AtenVarDimOp>(op, rewriter))) {
      return rewriter.notifyMatchFailure(op, "invalid variance parameters");
    }
    return success();
  }
};

struct DecomposeOnTorchPass
    : public DecomposeOnTorchBase<DecomposeOnTorchPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<DecomposeAtenVarDimOp>(context);

    LogicalResult result =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(result)) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createDecomposeOnTorch() {
  return std::make_unique<DecomposeOnTorchPass>();
}
