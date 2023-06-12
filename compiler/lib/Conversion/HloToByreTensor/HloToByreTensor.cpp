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
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
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

byre::ComputeOp replaceMhloOpWithByreComputeOp(RewriterBase &rewriter,
                                               Operation *op,
                                               StringRef calleeName,
                                               ValueRange newOperands,
                                               bool appendArgTypes) {
  auto key =
      byre::getByreKey(calleeName,
                       llvm::to_vector(llvm::map_range(
                           newOperands, [](Value v) { return v.getType(); })),
                       op->getResultTypes(), appendArgTypes);
  return rewriter.replaceOpWithNewOp<byre::ComputeOp>(
      op, op->getResultTypes(), key, newOperands,
      /*memEffects*/ ArrayAttr{});
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

    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, iter->second, adaptor.getOperands(), appendArgTypes);

    if constexpr (keepAttrs) {
      addAttrs(computeOp.getOperation(), op->getAttrs());
    } else {
      static_cast<void>(computeOp);
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

    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, op.getCallTargetName(), adaptor.getOperands(), false);

    auto dictAttr =
        op->template getAttrOfType<DictionaryAttr>(getCustomCallAttrName());
    if (dictAttr) {
      NamedAttrList originAttrs = computeOp->getAttrs();
      originAttrs.append(dictAttr);
      computeOp->setAttrs(originAttrs);
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

class ConvertReshapeOp : public OpConversionPattern<mhlo::ReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ReshapeOp op, mhlo::ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operand = adaptor.getOperand();
    auto operandType = operand.getType().cast<ShapedType>();
    auto resultType = op.getType().cast<ShapedType>();

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
    auto argType = adaptor.getOperands()[0].getType().dyn_cast<ShapedType>();
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

class ConvertGatherOpToByrePattern
    : public OpConversionPattern<mhlo::GatherOp> {
public:
  ConvertGatherOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::GatherOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mhlo::GatherOp op, typename mhlo::GatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto startIndices = op.getStartIndices();
    auto startIndicesTy = startIndices.getType().cast<ShapedType>();
    if (!startIndicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked start_indices");
    }

    auto operand = op.getOperand();
    auto operandTy = operand.getType().cast<ShapedType>();
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

    auto resultTy = op.getResult().getType().dyn_cast<ShapedType>();
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

    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, "IndexSelectOp", adaptor.getOperands(), appendArgTypes);

    // FIXME: currently only support select starting from 0
    computeOp->setAttr("dim", rewriter.getI32IntegerAttr(0));

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

    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, "IndexPutOp", adaptor.getOperands(), appendArgTypes);

    // FIXME: currently only support select on dim0
    computeOp->setAttr("dim", rewriter.getI32IntegerAttr(0));

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
    if (adaptor.getLhs().getType().cast<ShapedType>().getRank() != 2 ||
        adaptor.getRhs().getType().cast<ShapedType>().getRank() != 2)
      return failure();

    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, "MatmulOp", adaptor.getOperands(), appendArgTypes);

    computeOp->setAttr("lhs_contracting_dimension",
                       rewriter.getI64IntegerAttr(1));
    computeOp->setAttr("rhs_contracting_dimension",
                       rewriter.getI64IntegerAttr(0));
    return success();
  }

private:
  bool appendArgTypes;
};

class ConvertDotGeneralOpToByrePattern
    : public OpConversionPattern<mhlo::DotGeneralOp> {
public:
  ConvertDotGeneralOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mhlo::DotGeneralOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mlir::mhlo::DotGeneralOp op,
                  mlir::mhlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dotDimensionNumbers = adaptor.getDotDimensionNumbers();
    assert(dotDimensionNumbers.getLhsContractingDimensions().size() == 1);
    assert(dotDimensionNumbers.getRhsContractingDimensions().size() == 1);
    if (dotDimensionNumbers.getLhsBatchingDimensions().size() == 0) {

      auto computeOp = replaceMhloOpWithByreComputeOp(
          rewriter, op, "MatmulOp", adaptor.getOperands(), appendArgTypes);

      // append attribute 'lhsContractingDimension' and
      // 'rhsContractingDimension'
      int64_t lhsContractingDimension =
          dotDimensionNumbers.getLhsContractingDimensions()[0];
      int64_t rhsContractingDimension =
          dotDimensionNumbers.getRhsContractingDimensions()[0];
      computeOp->setAttr("lhs_contracting_dimension",
                         rewriter.getI64IntegerAttr(lhsContractingDimension));
      computeOp->setAttr("rhs_contracting_dimension",
                         rewriter.getI64IntegerAttr(rhsContractingDimension));
    } else {
      // convert to BatchMatmulOp
      SmallVector<int64_t> batchingDimensions;
      for (int64_t i = 0,
                   e = op->getResult(0).getType().cast<ShapedType>().getRank();
           i < e - 2; i++) {
        batchingDimensions.push_back(i);
      }
      if (!dotDimensionNumbers.getLhsBatchingDimensions().equals(
              batchingDimensions) ||
          !dotDimensionNumbers.getRhsBatchingDimensions().equals(
              batchingDimensions)) {
        return op->emitOpError()
               << "can not handle unregular batching_dimensions";
      }

      auto computeOp = replaceMhloOpWithByreComputeOp(
          rewriter, op, "BatchMatmulOp", adaptor.getOperands(), appendArgTypes);

      // append attributes of batching and contracting dimensions
      int64_t lhsContractingDimension =
          dotDimensionNumbers.getLhsContractingDimensions()[0];
      int64_t rhsContractingDimension =
          dotDimensionNumbers.getRhsContractingDimensions()[0];
      auto lhsBatchingDimensions =
          dotDimensionNumbers.getLhsBatchingDimensions();
      auto rhsBatchingDimensions =
          dotDimensionNumbers.getRhsBatchingDimensions();
      computeOp->setAttr("lhs_contracting_dimension",
                         rewriter.getI64IntegerAttr(lhsContractingDimension));
      computeOp->setAttr("rhs_contracting_dimension",
                         rewriter.getI64IntegerAttr(rhsContractingDimension));
      computeOp->setAttr("lhs_batching_dimensions",
                         rewriter.getI64ArrayAttr(lhsBatchingDimensions));
      computeOp->setAttr("rhs_batching_dimensions",
                         rewriter.getI64ArrayAttr(rhsBatchingDimensions));
    }
    return success();
  }

private:
  bool appendArgTypes;
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
    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, "ConvOp", adaptor.getOperands(), appendArgTypes);
    addAttrs(computeOp.getOperation(), attrs.getAttrs());
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
    auto &block = region.front();
    if (block.getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(op, "unsupported block in reduce");
    }
    // check block args
    if (block.getNumArguments() != 2 ||
        !block.getArgument(0).getType().isa<TensorType>() ||
        !block.getArgument(1).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported block's arg in reduce");
    }

    // check block body
    auto retOp = block.getTerminator();
    if (!isa<mlir::mhlo::ReturnOp>(retOp) || retOp->getNumOperands() != 1 ||
        !retOp->getOperand(0).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported terminator in reduce's block");
    }

    auto reduceComputation = &block.front();
    std::string reduceOp;
    auto checkInitialValue = [&](auto &&checker) {
      Attribute constAttr;
      if (matchPattern(adaptor.getInitValues()[0], m_Constant(&constAttr))) {
        if (checker(constAttr))
          return success();
      }
      return rewriter.notifyMatchFailure(
          op, "unsupported initial value of reduce op");
    };

    // TODO: more reduceOp supported
    auto status =
        llvm::TypeSwitch<Operation *, LogicalResult>(reduceComputation)
            .Case<mhlo::AddOp>([&](...) {
              reduceOp = "ReduceSumOp";
              return checkInitialValue(isZeroAttribute);
            })
            .Case<mhlo::MaxOp>([&](...) {
              reduceOp = "ReduceMaxOp";
              return checkInitialValue(isMinValueAttribute);
            })
            .Default([&](...) {
              return rewriter.notifyMatchFailure(
                  op, "unsupported ops in reduce_computation in reduce");
            });
    if (failed(status))
      return status;

    if (reduceComputation->getOperand(0) != block.getArgument(0) ||
        reduceComputation->getOperand(1) != block.getArgument(1) ||
        reduceComputation->getResult(0) != retOp->getOperand(0)) {
      return rewriter.notifyMatchFailure(op, "invalid block body");
    }

    auto inputShape = adaptor.getInputs()[0].getType().dyn_cast<ShapedType>();
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

    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, reduceOp, adaptor.getInputs(), appendArgTypes);

    computeOp->setAttr("dimensions", op.getDimensionsAttr());

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
    // check whether reduce supported
    Region &region = op.getBody();
    // only support single block
    if (region.getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported region in reduce_window");
    }
    auto &block = region.front();
    if (block.getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported block in reduce_window");
    }
    // check block args
    if (block.getNumArguments() != 2 ||
        !block.getArgument(0).getType().isa<TensorType>() ||
        !block.getArgument(1).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block's arg in reduce_window");
    }

    // check block body
    auto retOp = block.getTerminator();
    if (!isa<mlir::mhlo::ReturnOp>(retOp) || retOp->getNumOperands() != 1 ||
        !retOp->getOperand(0).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported terminator in reduce's block");
    }

    Operation *reduceComputation = &block.front();

    // only support ReduceWindowMax now
    if (!isa<mhlo::MaxOp>(reduceComputation)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported ops in reduce_computation of reduce_window");
    }
    // TODO: more ReduceOp supported
    std::string reduceWinOp = "PoolMaxOp";
    Attribute constAttr;
    if (!matchPattern(adaptor.getInitValues()[0], m_Constant(&constAttr))) {
      return rewriter.notifyMatchFailure(op, "non-const initial value");
    }
    if (!isMinValueAttribute(constAttr)) {
      return rewriter.notifyMatchFailure(op, "unsupported initial value");
    }

    if (reduceComputation->getOperand(0) != block.getArgument(0) ||
        reduceComputation->getOperand(1) != block.getArgument(1) ||
        reduceComputation->getResult(0) != retOp->getOperand(0)) {
      return rewriter.notifyMatchFailure(op, "invalid block body");
    }

    auto inputShape = adaptor.getInputs()[0].getType().dyn_cast<ShapedType>();
    if (!inputShape || !inputShape.hasRank()) {
      return rewriter.notifyMatchFailure(op, "invalid input type");
    }

    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, reduceWinOp, adaptor.getInputs(), appendArgTypes);

    for (auto attr : op->getAttrs()) {
      computeOp->setAttr(attr.getName(), attr.getValue());
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
    if (selectBlock.getOperations().size() != 2 ||
        !isa<mlir::mhlo::ReturnOp>(selectBlock.getTerminator())) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block in select of select_and_scatter");
    }

    if (selectBlock.getNumArguments() != 2 ||
        !selectBlock.getArgument(0).getType().isa<TensorType>() ||
        !selectBlock.getArgument(1).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block's arg in select of select_and_scatter");
    }

    auto &scatterBlock = op.getScatter().front();
    if (scatterBlock.getOperations().size() != 2 ||
        !isa<mlir::mhlo::ReturnOp>(scatterBlock.getTerminator())) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block in scatter of select_and_scatter");
    }

    if (scatterBlock.getNumArguments() != 2 ||
        !scatterBlock.getArgument(0).getType().isa<TensorType>() ||
        !scatterBlock.getArgument(1).getType().isa<TensorType>()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block's arg in scatter of select_and_scatter");
    }

    // check whether valid PoolingGrad
    // only support MaxPoolingGrad now
    if (auto compare = dyn_cast<mhlo::CompareOp>(selectBlock.front())) {
      if (compare.getComparisonDirection() != mhlo::ComparisonDirection::GE ||
          compare->getOperand(0) != selectBlock.getArgument(0) ||
          compare->getOperand(1) != selectBlock.getArgument(1)) {
        return rewriter.notifyMatchFailure(
            op,
            "unsupported comparison_direction in select of select_and_scatter");
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported ops in select of select_and_scatter");
    }

    if (!isa<mhlo::AddOp>(scatterBlock.front()) ||
        scatterBlock.front().getOperand(0) != scatterBlock.getArgument(0) ||
        scatterBlock.front().getOperand(1) != scatterBlock.getArgument(1)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported ops in scatter of select_and_scatter");
    }

    // TODO: more SelectAndScatterOp supported
    std::string poolingGradOp = "PoolMaxGradOp";
    auto computeOp = replaceMhloOpWithByreComputeOp(
        rewriter, op, poolingGradOp,
        ValueRange{adaptor.getOperand(), adaptor.getSource()}, appendArgTypes);

    addAttrs(computeOp, op->getAttrs());
    return success();
  }
};

struct ConvertHloToByreTensorPass
    : public ConvertHloToByreTensorBase<ConvertHloToByreTensorPass> {
public:
  ConvertHloToByreTensorPass(bool appendArgTypes)
      : ConvertHloToByreTensorBase() {
    this->appendArgTypes = appendArgTypes;

    supportMap.insert({"mhlo.add", "AddOp"});
    supportMap.insert({"mhlo.transpose", "TransposeOp"});
    supportMap.insert({"mhlo.convert", "Typecvt"});
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateHloToByreTensorPattern(patterns, supportMap, appendArgTypes);
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<tensor::TensorDialect, byre::ByreDialect,
                           arith::ArithDialect>();

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
    const llvm::StringMap<llvm::StringRef> &supportMap, bool appendArgTypes) {

  patterns.add<ConvertToByrePattern<mhlo::AddOp>,
               ConvertToByrePattern<mhlo::ConvertOp>,
               ConvertToByrePattern<mhlo::TransposeOp, /*keepAttrs*/ true>>(
      patterns.getContext(), supportMap, appendArgTypes);

  patterns.add<ConvertCustomCallOpToByrePattern<mhlo::CustomCallOp>,
               ConvertCustomCallOpToByrePattern<ace::CustomCallOp>,
               ConvertGatherOpToByrePattern, ConvertScatterOpToByrePattern,
               ConvertDotOpToByrePattern, ConvertDotGeneralOpToByrePattern,
               ConvertConvOpToByrePattern, ConvertReduceOpToByrePattern,
               ConvertReduceWindowOpToByrePattern,
               ConvertSelectAndScatterOpToByrePattern>(patterns.getContext(),
                                                       appendArgTypes);

  patterns
      .add<ConvertConstLikeOp<mhlo::ConstantOp>,
           ConvertConstLikeOp<ace::ConstOp>, ConvertReshapeOp, ConvertSliceOp>(
          patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertHloToByreTensorPass(bool appendArgTypes) {
  return std::make_unique<ConvertHloToByreTensorPass>(appendArgTypes);
}
