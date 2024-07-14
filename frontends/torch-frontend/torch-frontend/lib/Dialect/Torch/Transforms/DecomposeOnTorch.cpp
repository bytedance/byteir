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
#include "llvm/ADT/StringSet.h"

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
static LogicalResult calculateVariance(OpTy op, PatternRewriter &rewriter,
                                       bool unbiased, double correction) {
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
  Value constantFloatOne =
      rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
  Value meanAlongDims = rewriter.create<AtenMeanDimOp>(
      loc, meanDimResultType, self, dimList, /*keepDim=*/constantTrue,
      /*dtype=*/constantNone);
  Value subMean = rewriter.create<AtenSubTensorOp>(
      loc, inputTensorTy, self, meanAlongDims, constantFloatOne);
  Value square = rewriter.create<AtenSquareOp>(loc, inputTensorTy, subMean);

  if (!unbiased) {
    Value result =
        rewriter.create<AtenMeanDimOp>(loc, outputTensorType, square, dimList,
                                       keepDim, /*dtype=*/constantNone);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Divide the square sum by productDimSize - correction.
  Value squareSum = rewriter.create<AtenSumDimIntListOp>(
      loc, outputTensorType, square, dimList, keepDim, /*dtype=*/constantNone);

  // `productDimSize` is product of sizes of dimensions to be reduced.
  Value productDimSize =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  for (Value dim : dimListElements) {
    Value dimSize = rewriter.create<AtenSizeIntOp>(loc, self, dim);
    productDimSize =
        rewriter.create<AtenMulIntOp>(loc, productDimSize, dimSize);
  }
  productDimSize = rewriter.create<AtenFloatScalarOp>(loc, productDimSize);
  Value cstCorrection = rewriter.create<Torch::ConstantFloatOp>(
      loc, rewriter.getF64FloatAttr(correction));
  // The `correction` value should be less than or equal to `productDimSize +
  // 1`.
  if (!isAssumingStrictSymbolicShapes(rewriter)) {
    Value productDimSizePlusOne = rewriter.create<AtenAddOp>(
        loc, productDimSize.getType(), productDimSize, constantFloatOne);
    Value cond = rewriter.create<AtenGeFloatOp>(loc, productDimSizePlusOne,
                                                cstCorrection);
    rewriter.create<RuntimeAssertOp>(
        loc, cond,
        "correction value should be less than or equal to productDimSize + 1");
  }
  Value productDimSizeSubCorrection =
      rewriter.create<AtenSubFloatOp>(loc, productDimSize, cstCorrection);
  Value result = rewriter.create<AtenDivScalarOp>(
      loc, outputTensorType, squareSum, productDimSizeSubCorrection);
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
    double correction = unbiased ? 1.0 : 0.0;
    if (failed(calculateVariance<AtenVarDimOp>(op, rewriter, unbiased,
                                               correction))) {
      return rewriter.notifyMatchFailure(op, "invalid variance parameters");
    }
    return success();
  }
};

// Decompose aten.var(x, dims) into:
// sub = aten.sub(x, aten.mean(x, dims))
// square = aten.square(sub)
// out = aten.sum(square, dims) / (productDimSize - correction)
class DecomposeAtenVarCorrectionOp
    : public OpRewritePattern<AtenVarCorrectionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarCorrectionOp op,
                                PatternRewriter &rewriter) const override {
    int64_t correctionValInt;
    double correctionValFloat = 1.0;
    if (!isa<Torch::NoneType>(op.getCorrection().getType())) {
      if (isa<Torch::FloatType>(op.getCorrection().getType())) {
        if (!matchPattern(op.getCorrection(),
                          m_TorchConstantFloat(&correctionValFloat)))
          return rewriter.notifyMatchFailure(
              op, "Only support constant int or float correction value for "
                  "aten.var");
      } else if (isa<Torch::IntType>(op.getCorrection().getType())) {
        if (!matchPattern(op.getCorrection(),
                          m_TorchConstantInt(&correctionValInt)))
          return rewriter.notifyMatchFailure(
              op, "Only support constant int or float correction value for "
                  "aten.var");
        correctionValFloat = (double)correctionValInt;
      } else {
        return rewriter.notifyMatchFailure(
            op, "unimplemented: correction value should be only constant int "
                "or float for aten.var");
      }
    }

    bool unbiased = correctionValFloat == 0.0 ? false : true;
    if (failed(calculateVariance<AtenVarCorrectionOp>(op, rewriter, unbiased,
                                                      correctionValFloat)))
      return rewriter.notifyMatchFailure(op, "invalid variance parameters");
    return success();
  }
};

struct DecomposeOnTorchPass
    : public DecomposeOnTorchBase<DecomposeOnTorchPass> {
  DecomposeOnTorchPass(ArrayRef<std::string> legalOps) {
    this->legalOps = legalOps;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    legalOpsSet.clear();
    legalOpsSet.insert(legalOps.begin(), legalOps.end());

    RewritePatternSet patterns(context);
    if (!legalOpsSet.contains("aten.var.dim")) {
      patterns.add<DecomposeAtenVarDimOp>(context);
    }
    if (!legalOpsSet.contains("aten.var.correction")) {
      patterns.add<DecomposeAtenVarCorrectionOp>(context);
    }

    LogicalResult result =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(result)) {
      signalPassFailure();
    }
  }

  llvm::StringSet<> legalOpsSet;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createDecomposeOnTorch(ArrayRef<std::string> legalOps) {
  return std::make_unique<DecomposeOnTorchPass>(legalOps);
}
