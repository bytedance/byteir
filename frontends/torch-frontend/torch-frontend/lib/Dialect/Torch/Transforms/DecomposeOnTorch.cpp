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

class DecomposeAtenScaledDotProductAttentionOp
    : public OpRewritePattern<AtenScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<AtenScaledDotProductAttentionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value query = op.getQuery();
    auto queryTy = cast<BaseTensorType>(query.getType());
    Value key = op.getKey();
    auto keyTy = cast<BaseTensorType>(key.getType());
    Value val = op.getValue();
    auto valTy = cast<BaseTensorType>(val.getType());
    auto resTy = cast<BaseTensorType>(op.getType());

    if (!queryTy.hasDtype() || !keyTy.hasDtype() || !valTy.hasDtype() ||
        !resTy.hasDtype())
      return op.emitError("Types of Q, K, V and result "
                          "are expected to have dtype.");

    if (!queryTy.hasSizes() || !keyTy.hasSizes() || !valTy.hasSizes() ||
        !resTy.hasSizes())
      return op.emitError("Types of Q, K, V and result "
                          "are expected to have shape.");

    if (!isa<mlir::FloatType>(queryTy.getDtype()) ||
        !isa<mlir::FloatType>(keyTy.getDtype()) ||
        !isa<mlir::FloatType>(valTy.getDtype()) ||
        !isa<mlir::FloatType>(resTy.getDtype()))
      return op.emitError("Q, K, V and result "
                          "are expected to have float dtype.");

    bool isCausal = false;
    if (!matchPattern(op.getIsCausal(), m_TorchConstantBool(&isCausal)))
      return op.emitError("is_causal must be a Scalar constant");

    double dropoutP;
    if (!matchPattern(op.getDropoutP(), m_TorchConstantFloat(&dropoutP)))
      return op.emitError("dropout_p must be a Scalar constant");

    if (dropoutP != 0.0f)
      return op.emitError("Dropout is NOT supported");

    Value mask = op.getAttnMask();
    auto maskTy = dyn_cast<BaseTensorType>(mask.getType());
    if (maskTy) {
      auto maskDty = maskTy.getOptionalDtype();
      if (!maskTy.hasDtype() ||
          (!isa<mlir::FloatType>(maskDty) && !maskDty.isSignlessInteger(1))) {
        return op.emitError("attn_mask must be a tensor of "
                            "boolean or float");
      }
      if (!maskTy.hasSizes()) {
        return op.emitError("attn_mask must have shape");
      }
      if (isCausal)
        return op.emitError("attn_mask and is_causal must be set exclusively.");
    }
    Value minusOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(-1));
    Value minusTwo = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(-2));
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value zero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);

    SmallVector<int64_t, 6> transShape(keyTy.getSizes());
    std::swap(transShape.end()[-1], transShape.end()[-2]);
    auto transTy = keyTy.getWithSizesAndDtype(llvm::ArrayRef(transShape),
                                              keyTy.getOptionalDtype());
    Value transTensor = rewriter.create<AtenTransposeIntOp>(loc, transTy, key,
                                                            minusOne, minusTwo);

    SmallVector<int64_t, 6> qkShape(resTy.getSizes());
    qkShape.end()[-2] = queryTy.getSizes().end()[-2];
    qkShape.end()[-1] = transShape.end()[-1];
    auto qkTy = cast<BaseTensorType>(queryTy.getWithSizesAndDtype(
        llvm::ArrayRef(qkShape), queryTy.getDtype()));
    Value qkTensor =
        rewriter.create<AtenMatmulOp>(loc, qkTy, query, transTensor);

    Value scale = op.getScale();
    auto scaleTy = dyn_cast<mlir::FloatType>(scale.getType());
    if (!scaleTy) {
      Value lastDimSizeOfQ =
          rewriter.create<AtenSizeIntOp>(loc, query, minusOne);
      Value sqrtVal = rewriter.create<AtenSqrtIntOp>(loc, lastDimSizeOfQ);
      scale = rewriter.create<AtenDivOp>(loc, one, sqrtVal);
    }

    Value scaledQKTensor =
        rewriter.create<AtenMulScalarOp>(loc, qkTy, qkTensor, scale);

    Value maskedTensor = scaledQKTensor;
    if (maskTy && isa<mlir::FloatType>(maskTy.getDtype())) {
      maskedTensor = rewriter.create<AtenAddTensorOp>(loc, qkTy, scaledQKTensor,
                                                      mask, one);
    } else if (maskTy || isCausal) {
      Value firstDimSizeOfMask =
          rewriter.create<AtenSizeIntOp>(loc, qkTensor, minusTwo);
      Value secondDimSizeOfMask =
          rewriter.create<AtenSizeIntOp>(loc, qkTensor, minusOne);
      Value dimList = rewriter.create<PrimListConstructOp>(
          loc, Torch::ListType::get(firstDimSizeOfMask.getType()),
          ValueRange({firstDimSizeOfMask, secondDimSizeOfMask}));
      Value dtypeInt = rewriter.create<PrimDtypeOp>(loc, scaledQKTensor);
      auto zerosTy = cast<BaseTensorType>(qkTy.getWithSizesAndDtype(
          llvm::ArrayRef<int64_t>{qkShape.end()[-2], qkShape.end()[-1]},
          qkTy.getDtype()));
      Value zeros = rewriter.create<AtenZerosOp>(loc, zerosTy, dimList,
                                                 dtypeInt, none, none, none);
      if (isCausal) {
        auto noneSizeBoolType =
            queryTy.getWithSizesAndDtype(std::nullopt, rewriter.getI1Type());
        Value ones = rewriter.create<AtenOnesOp>(
            loc, noneSizeBoolType, dimList,
            getDtypeIntValueForType(rewriter, loc, rewriter.getI1Type()), none,
            none, none);
        mask = rewriter.create<AtenTrilOp>(loc, ones.getType(), ones, zero);
      }
      maskTy = cast<BaseTensorType>(mask.getType());

      Value notMask = rewriter.create<AtenLogicalNotOp>(loc, maskTy, mask);
      auto dType = cast<mlir::FloatType>(queryTy.getDtype());
      Value minimalVal = rewriter.create<Torch::ConstantFloatOp>(
          loc, rewriter.getFloatAttr(
                   rewriter.getF64Type(),
                   llvm::APFloat::getInf(dType.getFloatSemantics(), true)
                       .convertToDouble()));

      mask = rewriter.create<AtenMaskedFillScalarOp>(
          loc,
          maskTy.getWithSizesAndDtype(maskTy.getSizes(), zerosTy.getDtype()),
          zeros, notMask, minimalVal);
      maskedTensor = rewriter.create<AtenAddTensorOp>(loc, qkTy, scaledQKTensor,
                                                      mask, one);
    }

    Value softmaxTensor = rewriter.create<AtenSoftmaxIntOp>(
        loc, qkTy, maskedTensor, minusOne, none);

    rewriter.replaceOpWithNewOp<AtenMatmulOp>(op, resTy, softmaxTensor, val);
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
    if (!legalOpsSet.contains("aten.scaled_dot_product_attention")) {
      patterns.add<DecomposeAtenScaledDotProductAttentionOp>(context);
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
