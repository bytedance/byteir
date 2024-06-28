//===- HloToByreTensor.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/HloToByreTensor/HloToByreTensor.h"
#include "byteir/Conversion/FuncToByre/FuncToByre.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

FailureOr<byre::ComputeOnTensorOp> replaceMhloOpWithByreComputeOnTensorOp(
    RewriterBase &rewriter, Operation *op, StringRef calleeName,
    ValueRange newOperands, bool appendArgTypes) {
  auto key =
      byre::getByreKey(calleeName,
                       llvm::to_vector(llvm::map_range(
                           newOperands, [](Value v) { return v.getType(); })),
                       op->getResultTypes(), appendArgTypes);
  auto emptyTensors = createEmptyTensorForOpResult(rewriter, op);
  if (failed(emptyTensors)) {
    return failure();
  }

  return rewriter.replaceOpWithNewOp<byre::ComputeOnTensorOp>(
      op, op->getResultTypes(), key, newOperands, *emptyTensors);
}

template <typename SrcOpTy, bool keepAttrs = false>
class ConvertToByrePattern : public OpConversionPattern<SrcOpTy> {
public:
  ConvertToByrePattern(MLIRContext *ctx, const llvm::StringMap<StringRef> &lut,
                       bool appendTypes)
      : OpConversionPattern<SrcOpTy>(ctx), srcToCallee(lut),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto &&iter = srcToCallee.find(op.getOperation()->getName().getStringRef());
    if (iter == srcToCallee.end()) {
      // TODO adding more error message
      return failure();
    }

    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, iter->second, adaptor.getOperands(), appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    if constexpr (keepAttrs) {
      addAttrs(computeOnTensorOp.getOperation(), op->getAttrs());
    } else {
      static_cast<void>(computeOnTensorOp);
    }

    return success();
  }

protected:
  const llvm::StringMap<StringRef> &srcToCallee;
  bool appendArgTypes;
};

template <typename CustomCallOp>
class ConvertCustomCallOpToByrePattern
    : public OpConversionPattern<CustomCallOp> {
public:
  ConvertCustomCallOpToByrePattern(MLIRContext *ctx, bool /*appendArgTypes*/)
      : OpConversionPattern<CustomCallOp>(ctx) {}

  LogicalResult
  matchAndRewrite(CustomCallOp op, typename CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, op.getCallTargetName(), adaptor.getOperands(), false);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    auto dictAttr =
        op->template getAttrOfType<DictionaryAttr>(getCustomCallAttrName());
    if (dictAttr) {
      NamedAttrList originAttrs = computeOnTensorOp->getAttrs();
      originAttrs.append(dictAttr);
      computeOnTensorOp->setAttrs(originAttrs);
    }

    return success();
  }
};

template <typename ConstOp>
class ConvertConstLikeOp : public OpConversionPattern<ConstOp> {
public:
  using OpConversionPattern<ConstOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstOp constOp, typename ConstOp::Adaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.getValue());
    return success();
  }
};

template <typename OP> class ConvertReshapeOp : public OpConversionPattern<OP> {
public:
  using OpConversionPattern<OP>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OP op, typename OP::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operand = adaptor.getOperand();
    auto operandType = llvm::cast<ShapedType>(operand.getType());
    auto resultType = llvm::cast<ShapedType>(op.getType());

    if (!operandType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();

    if (auto reassociationMap =
            getReassociationIndicesForReshape(operandType, resultType)) {
      if (resultType.getRank() < operandType.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, resultType, operand, *reassociationMap);
      } else {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, resultType, operand, *reassociationMap);
      }
      return success();
    }

    Value newReshaped = operand;
    if (operandType.getRank() != 1) {
      if (auto reassociationIndices = getReassociationIndicesForCollapse(
              operandType.getShape(), {operandType.getNumElements()})) {
        newReshaped = rewriter.create<tensor::CollapseShapeOp>(
            op->getLoc(), newReshaped, *reassociationIndices);
      } else {
        return failure();
      }
    }

    if (resultType.getRank() != 1) {
      if (auto reassociationIndices = getReassociationIndicesForCollapse(
              resultType.getShape(), {resultType.getNumElements()})) {
        newReshaped = rewriter.create<tensor::ExpandShapeOp>(
            op->getLoc(), resultType, newReshaped, *reassociationIndices);
      } else {
        return failure();
      }
    }

    rewriter.replaceOp(op, newReshaped);
    return success();
  }
};

class ConvertSliceOp : public OpConversionPattern<mhlo::SliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::SliceOp sliceOp,
                  typename mhlo::SliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto argType = dyn_cast<ShapedType>(adaptor.getOperands()[0].getType());
    if (!argType || !argType.hasRank()) {
      return rewriter.notifyMatchFailure(sliceOp, "expects known-rank args");
    }

    SmallVector<OpFoldResult, 3> offsets, sizes, strides;
    for (int i = 0, e = argType.getRank(); i < e; ++i) {
      auto start = sliceOp.getStartIndices().getValues<int64_t>()[i];
      auto limit = sliceOp.getLimitIndices().getValues<int64_t>()[i];
      auto stride = sliceOp.getStrides().getValues<int64_t>()[i];
      offsets.push_back(rewriter.getI64IntegerAttr(start));
      // Say that there are k elements in total, we have condition:
      //   start + (k - 1) * strides <= limit - 1
      // ->
      //   k <= (limit - 1 - start + strides) / strides
      sizes.push_back(
          rewriter.getI64IntegerAttr((limit - 1 - start + stride) / stride));
      strides.push_back(rewriter.getI64IntegerAttr(stride));
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        sliceOp, adaptor.getOperands()[0], offsets, sizes, strides);
    return success();
  }
};

class ConvertConcatenateOp : public OpConversionPattern<mhlo::ConcatenateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ConcatenateOp concatOp,
                  typename mhlo::ConcatenateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // All input tensors must have the same shape, except in the cat
    // dimension. And all tensor must have static size in the cat dimension.
    auto resultType = concatOp.getType();

    uint64_t axis = concatOp.getDimension();
    if (llvm::any_of(adaptor.getOperands(), [&](auto &&value) {
          return cast<ShapedType>(value.getType()).isDynamicDim(axis);
        }))
      return failure();

    auto firstOperand = adaptor.getOperands()[0];
    SmallVector<Value> sizes;
    SmallVector<int64_t> static_offsets(resultType.getRank(), 0), static_sizes,
        static_strides(resultType.getRank(), 1);
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (resultType.isDynamicDim(i)) {
        static_sizes.push_back(ShapedType::kDynamic);
        sizes.push_back(rewriter.create<tensor::DimOp>(concatOp->getLoc(),
                                                       firstOperand, i));
      } else {
        static_sizes.push_back(resultType.getDimSize(i));
      }
    }

    SmallVector<Value> dynDims;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      if (resultType.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(concatOp->getLoc(),
                                                         firstOperand, i));
      }
    Value value = rewriter.create<tensor::EmptyOp>(concatOp->getLoc(),
                                                   resultType, dynDims);
    int64_t upperBound = 0;
    for (auto &&operand : adaptor.getOperands()) {
      auto operandType = cast<ShapedType>(operand.getType());
      static_offsets[axis] = upperBound;
      static_sizes[axis] = operandType.getDimSize(axis);
      value = rewriter.create<tensor::InsertSliceOp>(
          concatOp->getLoc(), operand, value, ValueRange(), sizes, ValueRange(),
          static_offsets, static_sizes, static_strides);
      upperBound += operandType.getDimSize(axis);
    }
    rewriter.replaceOp(concatOp, value);
    return success();
  }
};

class ConvertGatherOpToByrePattern
    : public OpConversionPattern<mhlo::GatherOp> {
public:
  ConvertGatherOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::GatherOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mhlo::GatherOp op, typename mhlo::GatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto startIndices = op.getStartIndices();
    auto startIndicesTy = cast<ShapedType>(startIndices.getType());
    if (!startIndicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked start_indices");
    }

    auto operand = op.getOperand();
    auto operandTy = cast<ShapedType>(operand.getType());
    if (!operandTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked operand");
    }

    int64_t indexVectorDim = startIndicesTy.getRank();

    auto dimensionNumbers = op.getDimensionNumbers();
    if (dimensionNumbers.getIndexVectorDim() != indexVectorDim) {
      return rewriter.notifyMatchFailure(
          op, "index_vector_dim not last dimension of start_indices");
    }

    // Index select only works across a single dimension.
    if (startIndicesTy.getShape().empty()) {
      return rewriter.notifyMatchFailure(
          op, "empty start_indices index vector dimension");
    }

    // Only support the default case for start_index_map.
    if (dimensionNumbers.getStartIndexMap().size() != 1 ||
        dimensionNumbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(op, "start_index_map != [0]");
    }

    auto resultTy = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "unranked result");
    }

    // Offset dimensions should be the defaults.
    if (dimensionNumbers.getOffsetDims().size() !=
        static_cast<size_t>(resultTy.getRank() - indexVectorDim)) {
      return rewriter.notifyMatchFailure(
          op, "offset_dims.size not operand rank minus index_vector_dim");
    }

    for (auto it : llvm::enumerate(dimensionNumbers.getOffsetDims())) {
      if ((it.index() + indexVectorDim) != static_cast<size_t>(it.value())) {
        return rewriter.notifyMatchFailure(
            op, "offset_dims != [index_vector_dim, result.rank)");
      }
    }

    for (auto it : llvm::enumerate(op.getSliceSizes().getValues<APInt>())) {
      // First shape value must be 1.
      if (it.index() == 0) {
        if (it.value().getSExtValue() != 1) {
          return rewriter.notifyMatchFailure(op, "slice_size[0] != 1");
        }
        continue;
      }

      // The op needs to index the entire slice for each other dimension.
      if (it.value().getSExtValue() != operandTy.getDimSize(it.index())) {
        return rewriter.notifyMatchFailure(
            op, "slice_size doesn't match operand dimension");
      }
    }

    if (dimensionNumbers.getCollapsedSliceDims().size() != 1 ||
        dimensionNumbers.getCollapsedSliceDims()[0] != 0) {
      return rewriter.notifyMatchFailure(op, "collapsed_slice_dims != [0]");
    }

    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, "IndexSelectOp", adaptor.getOperands(), appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    // FIXME: currently only support select starting from 0
    computeOnTensorOp->setAttr("dim", rewriter.getI32IntegerAttr(0));

    return success();
  }

private:
  bool appendArgTypes;
};

class ConvertScatterOpToByrePattern
    : public OpConversionPattern<mhlo::ScatterOp> {
public:
  ConvertScatterOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::ScatterOp>(ctx), appendArgTypes(appendTypes) {
  }

  LogicalResult
  matchAndRewrite(mhlo::ScatterOp op, typename mhlo::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // check wthether scatter supported
    Region &region = op.getUpdateComputation();
    // only support single block
    if (region.getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported region in scatter");
    }

    auto &block = region.front();
    if (!isBlockSingleOp<mhlo::AddOp>(&block)) {
      return rewriter.notifyMatchFailure(op, "unsupported block in scatter");
    }

    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, "IndexPutOp", adaptor.getOperands(), appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    // FIXME: currently only support select on dim0
    computeOnTensorOp->setAttr("dim", rewriter.getI32IntegerAttr(0));

    return success();
  }

private:
  bool appendArgTypes;
};

class ConvertDotOpToByrePattern : public OpConversionPattern<mhlo::DotOp> {
public:
  ConvertDotOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::DotOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mlir::mhlo::DotOp op, mlir::mhlo::DotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: support matrix * vector, vector * matrix and vector * vector
    if (cast<ShapedType>(adaptor.getLhs().getType()).getRank() != 2 ||
        cast<ShapedType>(adaptor.getRhs().getType()).getRank() != 2)
      return failure();

    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, "MatmulOp", adaptor.getOperands(), appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    computeOnTensorOp->setAttr("lhs_contracting_dimension",
                               rewriter.getI64IntegerAttr(1));
    computeOnTensorOp->setAttr("rhs_contracting_dimension",
                               rewriter.getI64IntegerAttr(0));
    return success();
  }

private:
  bool appendArgTypes;
};

class ConvertDotGeneralOpToByrePattern
    : public OpConversionPattern<mhlo::DotGeneralOp> {
public:
  ConvertDotGeneralOpToByrePattern(MLIRContext *ctx, bool appendTypes,
                                   bool enableTF32)
      : OpConversionPattern<mhlo::DotGeneralOp>(ctx),
        appendArgTypes(appendTypes), enableTF32(enableTF32) {}

  LogicalResult
  matchAndRewrite(mlir::mhlo::DotGeneralOp op,
                  mlir::mhlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dotDimensionNumbers = adaptor.getDotDimensionNumbers();
    if (dotDimensionNumbers.getLhsContractingDimensions().size() != 1) {
      return failure();
    }
    if (dotDimensionNumbers.getRhsContractingDimensions().size() != 1) {
      return failure();
    }
    auto lhsBatchs = dotDimensionNumbers.getLhsBatchingDimensions();
    auto rhsBatchs = dotDimensionNumbers.getRhsBatchingDimensions();
    size_t lhsRank = cast<ShapedType>(op.getLhs().getType()).getRank();
    size_t rhsRank = cast<ShapedType>(op.getRhs().getType()).getRank();
    if (lhsRank != rhsRank) {
      return failure();
    }
    if (lhsRank != lhsBatchs.size() + 2 || rhsRank != rhsBatchs.size() + 2) {
      return failure();
    }

    if (dotDimensionNumbers.getLhsBatchingDimensions().size() == 0 &&
        dotDimensionNumbers.getRhsBatchingDimensions().size() == 0) {
      // convert to MatmulOp
      auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
          rewriter, op, "MatmulOp", adaptor.getOperands(), appendArgTypes);
      if (failed(failureOrComputeOnTensorOp)) {
        return failure();
      }
      auto computeOnTensorOp = *failureOrComputeOnTensorOp;
      int64_t lhsContractingDimension =
          dotDimensionNumbers.getLhsContractingDimensions()[0];
      int64_t rhsContractingDimension =
          dotDimensionNumbers.getRhsContractingDimensions()[0];
      computeOnTensorOp->setAttr(
          "lhs_contracting_dimension",
          rewriter.getI64IntegerAttr(lhsContractingDimension));
      computeOnTensorOp->setAttr(
          "rhs_contracting_dimension",
          rewriter.getI64IntegerAttr(rhsContractingDimension));
      if (this->enableTF32) {
        computeOnTensorOp->setAttr("compute_type",
                                   TypeAttr::get(rewriter.getTF32Type()));
      }
    } else {
      // convert to BatchMatmulOp
      SmallVector<int64_t> batchingDimensions =
          to_vector(llvm::seq<int64_t>(0, lhsRank - 2));
      if (!dotDimensionNumbers.getLhsBatchingDimensions().equals(
              batchingDimensions) ||
          !dotDimensionNumbers.getRhsBatchingDimensions().equals(
              batchingDimensions)) {
        return op->emitOpError()
               << "can not handle unregular batching_dimensions";
      }

      auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
          rewriter, op, "BatchMatmulOp", adaptor.getOperands(), appendArgTypes);
      if (failed(failureOrComputeOnTensorOp)) {
        return failure();
      }
      auto computeOnTensorOp = *failureOrComputeOnTensorOp;
      // append attributes of batching and contracting dimensions
      int64_t lhsContractingDimension =
          dotDimensionNumbers.getLhsContractingDimensions()[0];
      int64_t rhsContractingDimension =
          dotDimensionNumbers.getRhsContractingDimensions()[0];
      auto lhsBatchingDimensions =
          dotDimensionNumbers.getLhsBatchingDimensions();
      auto rhsBatchingDimensions =
          dotDimensionNumbers.getRhsBatchingDimensions();
      computeOnTensorOp->setAttr(
          "lhs_contracting_dimension",
          rewriter.getI64IntegerAttr(lhsContractingDimension));
      computeOnTensorOp->setAttr(
          "rhs_contracting_dimension",
          rewriter.getI64IntegerAttr(rhsContractingDimension));
      computeOnTensorOp->setAttr(
          "lhs_batching_dimensions",
          rewriter.getI64ArrayAttr(lhsBatchingDimensions));
      computeOnTensorOp->setAttr(
          "rhs_batching_dimensions",
          rewriter.getI64ArrayAttr(rhsBatchingDimensions));
      if (this->enableTF32) {
        computeOnTensorOp->setAttr("compute_type",
                                   TypeAttr::get(rewriter.getTF32Type()));
      }
    }
    return success();
  }

private:
  bool appendArgTypes;
  bool enableTF32;
};

class ConvertConvOpToByrePattern
    : public OpConversionPattern<mlir::mhlo::ConvolutionOp> {
public:
  ConvertConvOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mlir::mhlo::ConvolutionOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mlir::mhlo::ConvolutionOp op,
                  mlir::mhlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    NamedAttrList attrs;
    handleConvAttribute(attrs, op, rewriter);
    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, "ConvOp", adaptor.getOperands(), appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    addAttrs(computeOnTensorOp.getOperation(), attrs.getAttrs());
    return success();
  }

private:
  bool appendArgTypes;
};

class ConvertReduceOpToByrePattern
    : public OpConversionPattern<mhlo::ReduceOp> {
private:
  bool appendArgTypes;

public:
  ConvertReduceOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::ReduceOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mhlo::ReduceOp op, mhlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 1 || op->getNumResults() != 1 ||
        adaptor.getInitValues().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "batched reductions is not supported yet");
    }
    // check whether reduce supported
    Region &region = op.getBody();
    // only support single block
    if (region.getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op, "unsupported region in reduce");
    }
    Attribute constAttr;
    if (!matchPattern(adaptor.getInitValues()[0], m_Constant(&constAttr))) {
      return rewriter.notifyMatchFailure(op, "non-const initial value");
    }

    auto &block = region.front();
    std::string reduceOp = "";
    if (isBlockSingleOp<mhlo::AddOp>(&block) && isZeroAttribute(constAttr)) {
      reduceOp = "ReduceSumOp";
    } else if (isBlockSingleOp<mhlo::MaxOp>(&block) &&
               isMinValueAttribute(constAttr)) {
      reduceOp = "ReduceMaxOp";
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported block in reduce");
    }

    auto inputShape = dyn_cast<ShapedType>(adaptor.getInputs()[0].getType());
    if (!inputShape || !inputShape.hasRank()) {
      return rewriter.notifyMatchFailure(op, "invalid input type");
    }

    std::vector<int64_t> dimensions;
    for (auto &&i : op.getDimensionsAttr()) {
      auto dim = i.getSExtValue();
      if (dim < 0 || dim >= inputShape.getRank()) {
        return rewriter.notifyMatchFailure(op, "invalid reduce dimensions");
      }
      dimensions.push_back(dim);
    }
    std::sort(dimensions.begin(), dimensions.end());
    for (size_t i = 0; i < dimensions.size() - 1; ++i) {
      if (dimensions[i + 1] - dimensions[i] != 1)
        return rewriter.notifyMatchFailure(
            op, "only consecutive dimensions were support");
    }

    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, reduceOp, adaptor.getInputs(), appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    computeOnTensorOp->setAttr("dimensions", op.getDimensionsAttr());

    return success();
  }
};

class ConvertReduceWindowOpToByrePattern
    : public OpConversionPattern<mhlo::ReduceWindowOp> {
private:
  bool appendArgTypes;

public:
  ConvertReduceWindowOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::ReduceWindowOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mhlo::ReduceWindowOp op,
                  mhlo::ReduceWindowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (adaptor.getInputs().size() != 1 || op->getNumResults() != 1 ||
        adaptor.getInitValues().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "batched reductions is not supported yet");
    }
    auto inputShape = dyn_cast<ShapedType>(adaptor.getInputs()[0].getType());
    if (!inputShape || !inputShape.hasRank()) {
      return rewriter.notifyMatchFailure(op, "invalid input type");
    }

    // check whether reduce supported
    Region &region = op.getBody();
    // only support single block
    if (region.getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported region in reduce_window");
    }
    Attribute constAttr;
    if (!matchPattern(adaptor.getInitValues()[0], m_Constant(&constAttr))) {
      return rewriter.notifyMatchFailure(op, "non-const initial value");
    }

    auto &block = region.front();
    std::string byrecomputeOnTensorOpName = "";
    if (isBlockSingleOp<mhlo::MaxOp>(&block) &&
        isMinValueAttribute(constAttr)) {
      byrecomputeOnTensorOpName = "PoolMaxOp";
    } else if (isBlockSingleOp<mhlo::AddOp>(&block) &&
               isZeroAttribute(constAttr)) {
      byrecomputeOnTensorOpName = "PoolSumOp";
    } else {
      return rewriter.notifyMatchFailure(op, "unsupport reduce_window");
    }

    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, byrecomputeOnTensorOpName, adaptor.getInputs(),
        appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    for (auto attr : op->getAttrs()) {
      computeOnTensorOp->setAttr(attr.getName(), attr.getValue());
    }

    return success();
  }
};

class ConvertSelectAndScatterOpToByrePattern
    : public OpConversionPattern<mhlo::SelectAndScatterOp> {
private:
  bool appendArgTypes;

public:
  ConvertSelectAndScatterOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::SelectAndScatterOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mhlo::SelectAndScatterOp op,
                  mhlo::SelectAndScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // check whether SelectAndScatterOp support
    if (op.getSelect().getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "unsupported select in select_and_scatter");
    }

    if (op.getScatter().getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "unsupported scatter in select_and_scatter");
    }

    auto &selectBlock = op.getSelect().front();
    if (!isBlockSingleOp<mhlo::CompareOp>(&selectBlock)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block in select of select_and_scatter");
    }
    // check whether valid PoolingGrad
    // only support MaxPoolingGrad now
    auto compare = cast<mhlo::CompareOp>(selectBlock.front());
    if (compare.getComparisonDirection() != mhlo::ComparisonDirection::GE ||
        compare->getOperand(0) != selectBlock.getArgument(0) ||
        compare->getOperand(1) != selectBlock.getArgument(1)) {
      return rewriter.notifyMatchFailure(
          op,
          "unsupported comparison_direction in select of select_and_scatter");
    }

    auto &scatterBlock = op.getScatter().front();
    if (!isBlockSingleOp<mhlo::AddOp>(&scatterBlock)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block in scatter of select_and_scatter");
    }

    // TODO: more SelectAndScatterOp supported
    std::string poolingGradOp = "PoolMaxGradOp";
    auto failureOrComputeOnTensorOp = replaceMhloOpWithByreComputeOnTensorOp(
        rewriter, op, poolingGradOp,
        ValueRange{adaptor.getOperand(), adaptor.getSource()}, appendArgTypes);
    if (failed(failureOrComputeOnTensorOp)) {
      return failure();
    }
    auto computeOnTensorOp = *failureOrComputeOnTensorOp;
    addAttrs(computeOnTensorOp, op->getAttrs());
    return success();
  }
};

struct ConvertHloToByreTensorPass
    : public ConvertHloToByreTensorBase<ConvertHloToByreTensorPass> {
public:
  ConvertHloToByreTensorPass(bool appendArgTypes, bool enableTF32)
      : ConvertHloToByreTensorBase() {
    this->appendArgTypes = appendArgTypes;
    this->enableTF32 = enableTF32;

    supportMap.insert({"mhlo.transpose", "TransposeOp"});
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateHloToByreTensorPattern(patterns, supportMap, appendArgTypes,
                                   enableTF32);
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<tensor::TensorDialect, byre::ByreDialect,
                           shape::ShapeDialect, arith::ArithDialect>();

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(funcOp, target, frozenPatterns))) {
      signalPassFailure();
    }
  }

private:
  llvm::StringMap<StringRef> supportMap;
};
}; // namespace

void mlir::populateHloToByreTensorPattern(
    RewritePatternSet &patterns,
    const llvm::StringMap<llvm::StringRef> &supportMap, bool appendArgTypes,
    bool enableTF32) {

  patterns.add<ConvertToByrePattern<mhlo::AddOp>,
               ConvertToByrePattern<mhlo::ConvertOp>,
               ConvertToByrePattern<mhlo::TransposeOp, /*keepAttrs*/ true>>(
      patterns.getContext(), supportMap, appendArgTypes);

  patterns.add<ConvertDotGeneralOpToByrePattern>(patterns.getContext(),
                                                 appendArgTypes, enableTF32);

  patterns.add<ConvertCustomCallOpToByrePattern<mhlo::CustomCallOp>,
               ConvertCustomCallOpToByrePattern<ace::CustomCallOp>,
               ConvertGatherOpToByrePattern, ConvertScatterOpToByrePattern,
               ConvertDotOpToByrePattern, ConvertConvOpToByrePattern,
               ConvertReduceOpToByrePattern, ConvertReduceWindowOpToByrePattern,
               ConvertSelectAndScatterOpToByrePattern>(patterns.getContext(),
                                                       appendArgTypes);

  patterns.add<
      ConvertConstLikeOp<mhlo::ConstantOp>, ConvertConstLikeOp<ace::ConstOp>,
      ConvertReshapeOp<mhlo::ReshapeOp>, ConvertReshapeOp<ace::ReshapeOp>,
      ConvertSliceOp, ConvertConcatenateOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertHloToByreTensorPass(bool appendArgTypes, bool enableTF32) {
  return std::make_unique<ConvertHloToByreTensorPass>(appendArgTypes,
                                                      enableTF32);
}
