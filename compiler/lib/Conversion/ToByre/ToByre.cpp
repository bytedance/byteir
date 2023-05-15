//===- ToByre.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/ToByre/ToByre.h"
#include "byteir/Conversion/Common/FunctionSupport.h"
#include "byteir/Conversion/ToByre/Common.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/HashUtils.h"
#include "byteir/Utils/Utils.h"
#include "lhlo/IR/lhlo_ops.h" // LmhloDialect
#include "mhlo/IR/hlo_ops.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>
#include <string>
#include <unordered_map>

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::lmhlo;
using namespace mlir::mhlo;
using namespace llvm;

namespace {
// TODO: move this to util if needed
bool isArgAlias(SmallVectorImpl<Value> &operands, Value src, Value dst) {
  bool is_arg_alias = false;
  // TODO: move this util
  // if output is an arg, swap in and out
  if (dst.getDefiningOp() == nullptr) {
    operands.push_back(dst);
    operands.push_back(src);
    is_arg_alias = true;
  } else if (src.getDefiningOp() == nullptr) {
    operands.push_back(src);
    operands.push_back(dst);
    is_arg_alias = true;
  } else {
    operands.push_back(src);
    operands.push_back(dst);
  }
  return is_arg_alias;
}
} // namespace

namespace mlir {
template <>
LogicalResult ConvertToByrePattern<lmhlo::GatherOp>::matchAndRewrite(
    lmhlo::GatherOp op, typename lmhlo::GatherOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
  if (found == srcToCallee.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

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

  auto resultTy = op.getOutput().getType().dyn_cast<ShapedType>();
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

  SmallVector<Type, 2> inputTypes{adaptor.getOperand().getType(),
                                  adaptor.getStartIndices().getType()};
  SmallVector<Type, 1> outputTypes{adaptor.getOutput().getType()};
  auto key = getByreKey(found->second, inputTypes, outputTypes, appendArgTypes);

  auto computeOp =
      replaceLmhloOpWithByreComputeOp(rewriter, op, key, adaptor.getOperands());

  // FIXME: currently only support select starting from 0
  SmallVector<int32_t> dimensions;
  dimensions.reserve(indexVectorDim);
  for (int32_t i = 0; i < indexVectorDim; ++i) {
    dimensions.push_back(i);
  }
  computeOp->setAttr("dimensions", rewriter.getI32TensorAttr(dimensions));

  return success();
}

template <>
LogicalResult ConvertToByrePattern<lmhlo::ScatterOp>::matchAndRewrite(
    lmhlo::ScatterOp op, typename lmhlo::ScatterOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
  if (found == srcToCallee.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

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

  SmallVector<Type, 3> inputTypes{adaptor.getOperand().getType(),
                                  adaptor.getScatterIndices().getType(),
                                  adaptor.getUpdates().getType()};
  SmallVector<Type, 1> outputTypes{adaptor.getOutput().getType()};
  auto key = getByreKey(found->second, inputTypes, outputTypes, appendArgTypes);

  // TODO support inplace
  auto newOp =
      replaceLmhloOpWithByreComputeOp(rewriter, op, key, adaptor.getOperands());

  // FIXME: currently only support select on dim0
  newOp->setAttr("dim", rewriter.getI32IntegerAttr(0));

  return success();
}

template <>
LogicalResult ConvertToByrePattern<lmhlo::SliceOp>::matchAndRewrite(
    lmhlo::SliceOp op, typename lmhlo::SliceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
  if (found == srcToCallee.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

  // check whether Slice is applicable for Alias
  if (!isSplatValue(op.getStrides(), 1)) {
    return rewriter.notifyMatchFailure(op, "unsupported strides of slice");
  }

  auto output = adaptor.getOperands()[1];
  auto shape = output.getType().cast<MemRefType>().getShape();
  auto startIndices = op.getStartIndices();
  int64_t numStart = startIndices.getNumElements();
  // check high dim of shape is 1
  if (numStart > 1) {
    for (int64_t i = 0; i < numStart - 1; ++i) {
      if (shape[i] != 1) {
        return rewriter.notifyMatchFailure(op, "unsupport shape of slice");
      }
    }
  }

  // get last element of startIndices
  int64_t lastStart = startIndices.getValues<int64_t>()[numStart - 1];

  // if output is an arg, use copy
  if (adaptor.getOperands()[1].getDefiningOp() == nullptr) {
    auto newOp = rewriter.replaceOpWithNewOp<byre::CopyOp>(
        op, adaptor.getOperands()[0], adaptor.getOperands()[1]);

    newOp->setAttr("offset", rewriter.getI32IntegerAttr(lastStart));
    return success();
  }

  auto newOp = replaceLmhloOpWithByreComputeOp(rewriter, op, found->second,
                                               adaptor.getOperands());

  newOp->setAttr("offset", rewriter.getI32IntegerAttr(lastStart));

  if (adaptor.getOperands()[0].getDefiningOp() == nullptr) {
    newOp->setAttr("arg_alias", rewriter.getUnitAttr());
  }

  return success();
}

template <>
LogicalResult ConvertToByrePattern<lmhlo::ReshapeOp>::matchAndRewrite(
    lmhlo::ReshapeOp op, typename lmhlo::ReshapeOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
  if (found == srcToCallee.end()) {
    return op->emitOpError() << "can not find matched byre_compute_name";
  }

  // If both args, replace it with copy
  if (adaptor.getOperands()[0].getDefiningOp() == nullptr &&
      adaptor.getOperands()[1].getDefiningOp() == nullptr) {
    rewriter.replaceOpWithNewOp<byre::CopyOp>(op, adaptor.getOperands()[0],
                                              adaptor.getOperands()[1]);

    return success();
  }

  SmallVector<Value, 2> operands;
  bool argAlias =
      isArgAlias(operands, adaptor.getOperands()[0], adaptor.getOperands()[1]);

  auto newOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
      op, found->second, ValueRange{operands[0]}, ValueRange{operands[1]});

  newOp->setAttr("offset", rewriter.getI32IntegerAttr(0));

  if (argAlias) {
    newOp->setAttr("arg_alias", rewriter.getUnitAttr());
  }

  return success();
}

} // namespace mlir

namespace {

class ConvertCallOpToByrePattern : public OpConversionPattern<func::CallOp> {
private:
  bool appendArgTypes;

public:
  ConvertCallOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<func::CallOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(func::CallOp op, func::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    func::FuncOp funcOp = getFuncOp(op);
    if (funcOp == nullptr) {
      return failure();
    }

    StringAttr nameAttr =
        funcOp->getAttrOfType<StringAttr>(byre::getByreComputeName());
    if (nameAttr == nullptr) {
      return failure();
    }

    bool effectiveAppendArgTypes =
        !funcOp->hasAttr(byre::getByreForceComputeNameAttrName()) &&
        appendArgTypes;

    // handle
    SmallVector<Value> operands;

    SmallVector<int64_t> offsets;
    ArrayAttr memoryEffectsAttr;
    auto readonlyOperandNum = op->getAttrOfType<IntegerAttr>(
        getByreCallOpReadonlyOperandNumAttrName());
    if (funcOp->hasAttr(getByreArgOffsetAttrName())) {
      auto offsetArray =
          funcOp->getAttrOfType<ArrayAttr>(getByreArgOffsetAttrName());

      offsets = llvm::to_vector(llvm::map_range(
          offsetArray.getAsRange<IntegerAttr>(),
          [&](IntegerAttr intAttr) { return intAttr.getInt(); }));

      for (auto offset : offsets) {
        operands.push_back(adaptor.getOperands()[offset]);
      }
      if (readonlyOperandNum) {
        memoryEffectsAttr = rewriter.getArrayAttr(llvm::to_vector(
            llvm::map_range(offsets, [&](auto offset) -> Attribute {
              if (offset < readonlyOperandNum.getInt()) {
                return rewriter.getAttr<MemoryEffectAttr>(MemoryEffect::Read);
              } else {
                return rewriter.getAttr<MemoryEffectAttr>(MemoryEffect::Write);
              }
            })));
      }
    } else {
      operands.insert(operands.end(), adaptor.getOperands().begin(),
                      adaptor.getOperands().end());
      if (readonlyOperandNum) {
        SmallVector<Attribute> memoryEffectAttrs;
        memoryEffectAttrs.append(
            readonlyOperandNum.getInt(),
            rewriter.getAttr<MemoryEffectAttr>(MemoryEffect::Read));
        memoryEffectAttrs.append(
            op->getNumOperands() - readonlyOperandNum.getInt(),
            rewriter.getAttr<MemoryEffectAttr>(MemoryEffect::Write));
        memoryEffectsAttr = rewriter.getArrayAttr(memoryEffectAttrs);
      }
    }
    SmallVector<Type> argTypes;
    for (auto val : funcOp.getArguments())
      argTypes.push_back(val.getType());
    auto resTypes = funcOp.getResultTypes();
    auto key = getByreKey(nameAttr.getValue(), argTypes, resTypes,
                          effectiveAppendArgTypes);

    mlir::byre::ComputeOp computeOp =
        rewriter.replaceOpWithNewOp<byre::ComputeOp>(op, key, operands,
                                                     memoryEffectsAttr);

    // copy byre attr, and remove prefix
    SmallVector<NamedAttribute> attrs;
    for (auto iter = funcOp->getAttrs().begin();
         iter != funcOp->getAttrs().end(); iter++) {
      if (byre::isByreComputeAttr(*iter)) {
        attrs.emplace_back(byre::removeByrePrefix(*iter));
      }
    }

    // handle arg-position sensitive attr here
    if (offsets.size() > 0) {
      // handle passthrough by inserting alias
      if (funcOp->hasAttr(getByrePassThroughArgAttrName())) {
        auto passThroughArray =
            funcOp->getAttrOfType<ArrayAttr>(getByrePassThroughArgAttrName());

        auto passThrough = llvm::to_vector(llvm::map_range(
            passThroughArray.getAsRange<IntegerAttr>(),
            [&](IntegerAttr intAttr) { return intAttr.getInt(); }));

        auto loc = op.getLoc();

        for (size_t i = 0; i < passThrough.size(); i += 2) {
          SmallVector<Value, 2> aliasOperands;
          Value dst = adaptor.getOperands()[passThrough[i]];
          Value src = adaptor.getOperands()[passThrough[i + 1]];

          if (auto alloc = dst.getDefiningOp<memref::AllocOp>()) {
            rewriter.replaceOpWithNewOp<byre::AliasOp>(alloc, alloc.getType(),
                                                       src, 0);
          } else if (auto alloc = dst.getDefiningOp<memref::AllocOp>()) {
            rewriter.replaceOpWithNewOp<byre::AliasOp>(alloc, alloc.getType(),
                                                       dst, 0);
          } else {
            // copy src to dst
            rewriter.create<byre::CopyOp>(loc, src, dst);
          }
        }
      }
    }

    addAttrs(computeOp.getOperation(), attrs);

    return success();
  }
};

class ConvertDotOpToByrePattern
    : public OpConversionPattern<mlir::lmhlo::DotOp> {
private:
  bool appendArgTypes;

public:
  ConvertDotOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mlir::lmhlo::DotOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mlir::lmhlo::DotOp op, mlir::lmhlo::DotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dotDimensionNumbers = adaptor.getDotDimensionNumbers();
    assert(dotDimensionNumbers.getLhsContractingDimensions().size() == 1);
    assert(dotDimensionNumbers.getRhsContractingDimensions().size() == 1);
    if (dotDimensionNumbers.getLhsBatchingDimensions().size() == 0) {
      // convert to MatmulOp
      SmallVector<Type, 2> inputTypes{adaptor.getLhs().getType(),
                                      adaptor.getRhs().getType()};
      SmallVector<Type, 1> outputTypes{adaptor.getOutput().getType()};
      auto key =
          getByreKey("MatmulOp", inputTypes, outputTypes, appendArgTypes);

      auto computeOp = replaceLmhloOpWithByreComputeOp(rewriter, op, key,
                                                       adaptor.getOperands());

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
                   e = op.getOutput().getType().cast<ShapedType>().getRank();
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

      SmallVector<Type, 2> inputTypes{adaptor.getLhs().getType(),
                                      adaptor.getRhs().getType()};
      SmallVector<Type, 1> outputTypes{adaptor.getOutput().getType()};
      auto key =
          getByreKey("BatchMatmulOp", inputTypes, outputTypes, appendArgTypes);
      auto computeOp = replaceLmhloOpWithByreComputeOp(rewriter, op, key,
                                                       adaptor.getOperands());

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
};

class ConvertConvOpToByrePattern
    : public OpConversionPattern<mlir::lmhlo::ConvolutionOp> {
private:
  bool appendArgTypes;

public:
  ConvertConvOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<mlir::lmhlo::ConvolutionOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(mlir::lmhlo::ConvolutionOp op,
                  mlir::lmhlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    NamedAttrList attrs;
    handleConvAttribute(attrs, op, rewriter);
    SmallVector<Type, 2> inputTypes{adaptor.getLhs().getType(),
                                    adaptor.getRhs().getType()};
    SmallVector<Type, 1> outputTypes{adaptor.getOutput().getType()};
    auto key = getByreKey("ConvOp", inputTypes, outputTypes, appendArgTypes);
    auto computeOp = replaceLmhloOpWithByreComputeOp(rewriter, op, key,
                                                     adaptor.getOperands());
    addAttrs(computeOp.getOperation(), attrs.getAttrs());
    return success();
  }
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
    mlir::DictionaryAttr dictAttr;
    std::optional<mlir::Attribute> backendConfig = op.getBackendConfig();

    if (backendConfig.has_value()) {
      // Support older API, StringAttr, for now
      if (auto backendConfigAsStringAttr =
              backendConfig->dyn_cast<mlir::StringAttr>()) {
        auto strref = backendConfigAsStringAttr.strref();
        if (!strref.empty()) {
          Attribute attrs = mlir::parseAttribute(strref, op->getContext());
          if (!attrs || !attrs.isa<mlir::DictionaryAttr>())
            return failure();
          dictAttr = attrs.cast<mlir::DictionaryAttr>();
        }
      }
    }

    auto computeOp = replaceLmhloOpWithByreComputeOp(
        rewriter, op, op.getCallTargetName(), adaptor.getOperands());
    if (dictAttr) {
      NamedAttrList originAttrs = computeOp->getAttrs();
      originAttrs.append(dictAttr);
      computeOp->setAttrs(originAttrs);
    }

    return success();
  }
};

class ConvertSelectAndScatterOpToByrePattern
    : public OpConversionPattern<lmhlo::SelectAndScatterOp> {
private:
  bool appendArgTypes;

public:
  ConvertSelectAndScatterOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<lmhlo::SelectAndScatterOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(lmhlo::SelectAndScatterOp op,
                  lmhlo::SelectAndScatterOp::Adaptor adaptor,
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

    SmallVector<Type, 2> inputTypes{adaptor.getOperand().getType(),
                                    adaptor.getSource().getType()};
    SmallVector<Type, 1> outputTypes{adaptor.getOut().getType()};
    auto key =
        getByreKey(poolingGradOp, inputTypes, outputTypes, appendArgTypes);

    auto computeOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
        op, key, ValueRange{adaptor.getOperand(), adaptor.getSource()},
        ValueRange{adaptor.getOut()});

    addAttrs(computeOp, op->getAttrs());
    return success();
  }
};

class ConvertReduceOpToByrePattern
    : public OpConversionPattern<lmhlo::ReduceOp> {
private:
  bool appendArgTypes;

public:
  ConvertReduceOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<lmhlo::ReduceOp>(ctx), appendArgTypes(appendTypes) {
  }

  LogicalResult
  matchAndRewrite(lmhlo::ReduceOp op, lmhlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInputs().size() != 1 || adaptor.getOut().size() != 1 ||
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
    if (block.getNumArguments() != 3 ||
        !block.getArgument(0).getType().isa<MemRefType>() ||
        !block.getArgument(1).getType().isa<MemRefType>() ||
        !block.getArgument(2).getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported block's arg in reduce");
    }

    // check block body
    auto retOp = block.getTerminator();
    if (!isa<mlir::lmhlo::TerminatorOp>(retOp)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported terminator in reduce's block");
    }

    auto reduceComputation = &block.front();
    std::string ReduceOp;
    auto checkInitialValue = [&](auto &&checker) {
      if (!llvm::any_of(
              adaptor.getInitValues()[0].getUses(), [&](OpOperand &use) {
                if (auto constOp = llvm::dyn_cast_or_null<lmhlo::ConstantOp>(
                        use.getOwner())) {
                  // for ReduceSum initial value must be zero
                  return checker(constOp.getValue());
                }
                return false;
              })) {
        return rewriter.notifyMatchFailure(
            op, "unsupported initial value of reduce op");
      }
      return success();
    };
    // TODO: more ReduceOp supported
    auto status =
        llvm::TypeSwitch<Operation *, LogicalResult>(reduceComputation)
            .Case<lmhlo::AddOp>([&](...) {
              ReduceOp = "ReduceSumOp";
              return checkInitialValue(isZeroAttribute);
            })
            .Case<lmhlo::MaxOp>([&](...) {
              ReduceOp = "ReduceMaxOp";
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
        reduceComputation->getOperand(2) != block.getArgument(2)) {
    }

    auto inputShape = adaptor.getInputs()[0].getType().dyn_cast<MemRefType>();
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

    SmallVector<Type, 1> inputTypes{adaptor.getInputs()[0].getType()};
    SmallVector<Type, 1> outputTypes{adaptor.getOut()[0].getType()};
    auto key = getByreKey(ReduceOp, inputTypes, outputTypes, appendArgTypes);

    auto computeOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
        op, key, ValueRange{adaptor.getInputs()[0]},
        ValueRange{adaptor.getOut()[0]});

    computeOp->setAttr("dimensions", op.getDimensionsAttr());

    return success();
  }
};

class ConvertReduceWindowOpToByrePattern
    : public OpConversionPattern<lmhlo::ReduceWindowOp> {
private:
  bool appendArgTypes;

public:
  ConvertReduceWindowOpToByrePattern(MLIRContext *ctx, bool appendTypes)
      : OpConversionPattern<lmhlo::ReduceWindowOp>(ctx),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(lmhlo::ReduceWindowOp op,
                  lmhlo::ReduceWindowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (adaptor.getInputs().size() != 1 || adaptor.getOut().size() != 1 ||
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
    if (block.getOperations().size() != 2 &&
        block.getOperations().size() != 4) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported block in reduce_window");
    }
    // check block args
    if (block.getNumArguments() != 3 ||
        !block.getArgument(0).getType().isa<MemRefType>() ||
        !block.getArgument(1).getType().isa<MemRefType>() ||
        !block.getArgument(2).getType().isa<MemRefType>()) {
      return rewriter.notifyMatchFailure(
          op, "unsupported block's arg in reduce_window");
    }

    // check block body
    auto retOp = block.getTerminator();
    if (!isa<mlir::lmhlo::TerminatorOp>(retOp)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported terminator in reduce's block");
    }

    Operation *reduceComputation = nullptr;
    if (block.getOperations().size() == 2) {
      reduceComputation = &block.front();
    } else {
      reduceComputation = block.front().getNextNode();
    }

    // only support ReduceWindowMax now
    if (!isa<lmhlo::MaxOp>(reduceComputation)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported ops in reduce_computation of reduce_window");
    }

    if (block.getOperations().size() == 2 &&
        (reduceComputation->getOperand(0) != block.getArgument(0) ||
         reduceComputation->getOperand(1) != block.getArgument(1) ||
         reduceComputation->getOperand(2) != block.getArgument(2))) {
      return rewriter.notifyMatchFailure(
          op, "unsupported ops in reduce_computation in reduce_window");
    }

    // TODO: more ReduceOp supported
    std::string ReduceWinOp = "PoolMaxOp";

    auto inputShape = adaptor.getInputs()[0].getType().dyn_cast<MemRefType>();
    if (!inputShape || !inputShape.hasRank()) {
      return rewriter.notifyMatchFailure(op, "invalid input type");
    }

    if (!llvm::any_of(adaptor.getInitValues()[0].getUses(), [](OpOperand &use) {
          if (auto constOp =
                  llvm::dyn_cast_or_null<lmhlo::ConstantOp>(use.getOwner())) {
            // for ReduceWindowsMax initial value must be minValue
            return isMinValueAttribute(constOp.getValue());
          }
          return false;
        })) {
      return failure();
    }
    SmallVector<Type, 1> inputTypes{adaptor.getInputs()[0].getType()};
    SmallVector<Type, 1> outputTypes{adaptor.getOut()[0].getType()};
    auto key = getByreKey(ReduceWinOp, inputTypes, outputTypes, appendArgTypes);

    auto computeOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
        op, key, ValueRange{adaptor.getInputs()[0]},
        ValueRange{adaptor.getOut()[0]});

    for (auto attr : op->getAttrs()) {
      computeOp->setAttr(attr.getName(), attr.getValue());
    }

    return success();
  }
};

template <typename ConstantOp>
class ConvertConstOpToByrePattern : public OpConversionPattern<ConstantOp> {
public:
  ConvertConstOpToByrePattern(MLIRContext *ctx, bool /*appendArgTypes*/)
      : OpConversionPattern<ConstantOp>(ctx) {}

  LogicalResult
  matchAndRewrite(ConstantOp op, typename ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // FIXME: only allow allocated memref for now
    auto alloc = op.getOutput().template getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    auto computeOp = replaceLmhloOpWithByreComputeOp(rewriter, op, "FillOp",
                                                     adaptor.getOperands());

    computeOp->setAttr("value", op.getValue());

    return success();
  }
};

class ConvertViewOpToByrePattern : public OpConversionPattern<memref::ViewOp> {
public:
  ConvertViewOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<memref::ViewOp>(ctx) {}

  LogicalResult
  matchAndRewrite(memref::ViewOp op, memref::ViewOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr offset;
    if (!matchPattern(adaptor.getByteShift(), m_Constant(&offset))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<byre::AliasOp>(
        op, op->getResult(0).getType(), adaptor.getSource(), offset.getInt());
    return success();
  }
};

class ConvertAliasLikeOpToByrePattern
    : public OpInterfaceConversionPattern<lace::AliasLikeOpInterface> {
public:
  using OpInterfaceConversionPattern<
      lace::AliasLikeOpInterface>::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(lace::AliasLikeOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstMemRefType =
        op->getResult(0).getType().dyn_cast_or_null<MemRefType>();
    if (!dstMemRefType)
      return failure();

    auto offsetInbytes =
        (op.getOffsetElem() * dstMemRefType.getElementTypeBitWidth() + 7) >> 3;

    rewriter.replaceOpWithNewOp<byre::AliasOp>(op, dstMemRefType, operands[0],
                                               offsetInbytes);

    return success();
  }
};

std::optional<StringAttr> getCalleeAttr(memref::CopyOp op) {
  auto ctx = op->getContext();
  auto srcSpace = op.getSource().getType().cast<MemRefType>().getMemorySpace();
  auto dstSpace = op.getTarget().getType().cast<MemRefType>().getMemorySpace();

  if (!srcSpace.isa_and_nonnull<StringAttr>() ||
      !dstSpace.isa_and_nonnull<StringAttr>()) {
    return std::nullopt;
  }

  auto srcRef = srcSpace.cast<StringAttr>().strref();
  auto dstRef = dstSpace.cast<StringAttr>().strref();
  return StringAttr::get(ctx, srcRef + "2" + dstRef);
}

class ConvertMemrefCopyOpToByrePattern
    : public OpConversionPattern<memref::CopyOp> {
public:
  ConvertMemrefCopyOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<memref::CopyOp>(ctx) {}

  LogicalResult
  matchAndRewrite(memref::CopyOp op, memref::CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newOp = rewriter.replaceOpWithNewOp<byre::CopyOp>(
        op, adaptor.getOperands()[0], adaptor.getOperands()[1]);

    auto maybeCallee = getCalleeAttr(op);

    if (maybeCallee.has_value()) {
      newOp->setAttr("callee", *maybeCallee);
    }

    return success();
  }
};

// Main Passes
struct ConvertToByrePass : public ConvertToByreBase<ConvertToByrePass> {
  ConvertToByrePass(bool appendArgTypes) : ConvertToByreBase() {
    this->appendArgTypes = appendArgTypes;
  }

  void runOnOperation() override;
};

struct ConvertFuncAndCallToByrePass
    : public ConvertFuncAndCallToByreBase<ConvertFuncAndCallToByrePass> {
  ConvertFuncAndCallToByrePass(bool appendArgTypes, bool removeDupOutputs)
      : ConvertFuncAndCallToByreBase() {
    this->appendArgTypes = appendArgTypes;
    this->removeDupOutputs = removeDupOutputs;

    // insert attrNames
    attrNames.push_back(byre::ByreDialect::getEntryPointFunctionAttrName());
    argAttrNames.push_back(
        byre::ByreDialect::getEntryPointFuncArgNameAttrName());
    argAttrNames.push_back(
        byre::ByreDialect::getEntryPointFuncArgTypeAttrName());
  }

  void runOnOperation() override;

  llvm::SmallVector<StringRef, 4> attrNames;
  llvm::SmallVector<StringRef, 4> argAttrNames;
  llvm::SmallVector<StringRef, 4> resultAttrNames;
};

struct ConvertLmhloToByrePass
    : public ConvertLmhloToByreBase<ConvertLmhloToByrePass> {
  ConvertLmhloToByrePass(bool appendArgTypes) : ConvertLmhloToByreBase() {
    this->appendArgTypes = appendArgTypes;

    // TODO: change to loading from outside
    lmhloSupportMap.insert({"lmhlo.add", "AddOp"});
    lmhloSupportMap.insert({"lmhlo.scatter", "IndexPutOp"});
    lmhloSupportMap.insert({"lmhlo.gather", "IndexSelectOp"});
    lmhloSupportMap.insert({"lmhlo.reshape", "AliasOp"});
    lmhloSupportMap.insert({"lmhlo.slice", "AliasOp"});
    lmhloSupportMap.insert({"lmhlo.transpose", "TransposeOp"});
    lmhloSupportMap.insert({"lmhlo.convert", "Typecvt"});
  }

  void runOnOperation() override;

  llvm::StringMap<StringRef> lmhloSupportMap;
};

static bool isFuncWithEntryPointPlaceholder(func::FuncOp func) {
  return func->hasAttr(
      getAttrPlaceholderName(ByreDialect::getEntryPointFunctionAttrName()));
}

static bool isEntryPointFunc(func::FuncOp func) {
  return func->hasAttr(ByreDialect::getEntryPointFunctionAttrName());
}

static bool isRewritablePrivateFunc(func::FuncOp func) {
  // check support attribute
  return func.isPrivate() && func->hasAttr(getByreComputeName());
}

// identify EntryPoint funciton
static void identifyEntryPointFuncAndCalls(
    ModuleOp m, llvm::SmallVector<func::FuncOp, 4> &entries,
    llvm::SmallVector<func::CallOp, 16> &calls,
    llvm::SmallVector<func::FuncOp, 16> &removeFuncs) {
  // get first entry func

  llvm::SmallPtrSet<Operation *, 16> callSet;

  for (auto func : m.getOps<func::FuncOp>()) {
    // skip non entry-point function or empty func
    if (!isFuncWithEntryPointPlaceholder(func) || func.isPrivate()) {
      continue;
    }
    entries.push_back(func);

    for (auto callOp : func.getOps<func::CallOp>()) {
      auto calleeFuncOp = getFuncOp(callOp);
      if (isRewritablePrivateFunc(calleeFuncOp) && !callSet.contains(callOp)) {
        calls.push_back(callOp);
        callSet.insert(callOp);
        removeFuncs.push_back(calleeFuncOp);
      }
    }
  }
}

static inline void relocateFuncOpResults(func::FuncOp func,
                                         bool removeDupOutputs) {
  unsigned idx = func.getNumArguments();
  replicateFuncOpResults(func, [&](func::ReturnOp retOp) {
    std::unordered_map<mlir::Operation *, mlir::BlockArgument> removeAllocOps;
    std::unordered_map<mlir::Value, unsigned, byteir::MlirValueHash>
        lmhloConstantValue;
    mlir::OpBuilder opBuilder(retOp);
    for (auto retValIter : llvm::enumerate(retOp.getOperands())) {
      auto retVal = retValIter.value();
      if (isLmhloConstantValue(retVal)) {
        // if return constant value, insert a memref.copy
        opBuilder.setInsertionPoint(retOp);
        opBuilder.create<memref::CopyOp>(
            retOp.getLoc(), retVal, func.getArgument(idx + retValIter.index()));
        if (lmhloConstantValue.find(retVal) == lmhloConstantValue.end()) {
          lmhloConstantValue[retVal] = idx + retValIter.index();
        } else {
          // append byre.arg_alias_index to func op
          func.setArgAttr(
              idx + retValIter.index(),
              ByreDialect::getEntryPointFuncArgAliasIndexAttrName(),
              opBuilder.getI64IntegerAttr(lmhloConstantValue[retVal]));
        }
      } else if (auto allocOp = retVal.getDefiningOp<memref::AllocOp>()) {
        if (removeAllocOps.find(allocOp.getOperation()) ==
            removeAllocOps.end()) {
          // add alloc op to remove list
          removeAllocOps[allocOp.getOperation()] =
              func.getArgument(idx + retValIter.index());
        } else if (removeDupOutputs) {
          assert(false && "Not implemented: remove dup function outputs");
        } else {
          // if not to remove dup memref.alloc values, insert a memref.copy
          opBuilder.setInsertionPoint(retOp);
          opBuilder.create<memref::CopyOp>(
              retOp.getLoc(), retVal,
              func.getArgument(idx + retValIter.index()));
          // append byre.arg_alias_index to func op
          func.setArgAttr(
              idx + retValIter.index(),
              ByreDialect::getEntryPointFuncArgAliasIndexAttrName(),
              opBuilder.getI64IntegerAttr(
                  removeAllocOps[allocOp.getOperation()].getArgNumber()));
        }
      } else if (retVal.isa<BlockArgument>()) {
        // if return value is input from entry function, insert a memref.copy
        opBuilder.setInsertionPoint(retOp);
        opBuilder.create<memref::CopyOp>(
            retOp.getLoc(), retVal, func.getArgument(idx + retValIter.index()));
        // append byre.argalias to func op
        func.setArgAttr(idx + retValIter.index(),
                        ByreDialect::getEntryPointFuncArgAliasIndexAttrName(),
                        opBuilder.getI64IntegerAttr(
                            retVal.cast<BlockArgument>().getArgNumber()));
      } else {
        // if return value not alloced in entry function (like alloced in inner
        // function), insert a memref.copy.
        opBuilder.setInsertionPoint(retOp);
        opBuilder.create<memref::CopyOp>(
            retOp.getLoc(), retVal, func.getArgument(idx + retValIter.index()));
      }
    }
    // replace alloc ops
    for (auto op : removeAllocOps) {
      auto value = op.first->getResult(0);
      value.replaceAllUsesWith(op.second);
      op.first->erase();
    }

    // build and remove return first
    opBuilder.setInsertionPoint(retOp);
    opBuilder.create<func::ReturnOp>(retOp.getLoc());
    retOp.erase();
  });
}

static inline void rewriteCallOpsForFuncOp(ArrayRef<func::CallOp> calls) {

  for (auto callOp : calls) {
    if (callOp.getNumResults() == 0) {
      continue;
    }
    mlir::OpBuilder opBuilder(callOp);
    SmallVector<Value, 4> oprands(callOp.getOperands());

    // change result to alloc
    for (auto r : callOp.getResults()) {
      auto alloc = opBuilder.create<memref::AllocOp>(
          callOp.getLoc(), r.getType().dyn_cast<MemRefType>());
      r.replaceAllUsesExcept(alloc.getResult(), callOp);
      oprands.push_back(alloc.getResult());
    }

    func::CallOp newCallOp = opBuilder.create<func::CallOp>(
        callOp.getLoc(), callOp.getCalleeAttr(), TypeRange(), oprands);
    newCallOp->setAttrs(callOp->getAttrs());
    // TODO : we assume that all arguments of the function is with
    // MemoryEffect::Read and all results of the function is with
    // MemoryEffect::Write, do we need a more accurate memory R/W analysis in
    // the function body?
    newCallOp->setAttr(getByreCallOpReadonlyOperandNumAttrName(),
                       opBuilder.getIndexAttr(callOp->getNumOperands()));
  }

  // remove all remove ops
  for (auto op : calls) {
    op->erase();
  }
}

static inline void markFuncOpInOutTypeForLmhlo(func::FuncOp func,
                                               unsigned inputCnt,
                                               unsigned outputCnt) {
  auto argTypeAttrName = byre::ByreDialect::getEntryPointFuncArgTypeAttrName();
  auto argNameAttrName = byre::ByreDialect::getEntryPointFuncArgNameAttrName();
  auto context = func->getContext();
  for (size_t idx = 0; idx < func.getNumArguments(); ++idx) {
    func.setArgAttr(
        idx, argNameAttrName,
        StringAttr::get(context, Twine("Input") + Twine(inputCnt++)));
    func.setArgAttr(idx, argTypeAttrName,
                    byre::EntryFuncArgTypeAttr::get(
                        context, byre::EntryFuncArgType::Input));
  }
  for (size_t idx = 0; idx < func.getNumResults(); ++idx) {
    func.setResultAttr(
        idx, argNameAttrName,
        StringAttr::get(context, Twine("Output") + Twine(outputCnt++)));
    func.setResultAttr(idx, argTypeAttrName,
                       byre::EntryFuncArgTypeAttr::get(
                           context, byre::EntryFuncArgType::Output));
  }
}

static inline void rewriteByreResultAttrsToFuncResultAttr(func::FuncOp func) {
  auto resultAttrsName = byre::ByreDialect::getEntryPointFuncResultAttrsName();
  removeAttrPlaceholders(func, {resultAttrsName});
  if (auto resultAttrs =
          func->getAttrOfType<mlir::ArrayAttr>(resultAttrsName)) {
    auto newResultAttrs = resultAttrs.getValue();
    if (func.getNumResults() != newResultAttrs.size())
      return;
    for (size_t i = 0; i < newResultAttrs.size(); ++i) {
      if (auto newResultAttrsDict =
              newResultAttrs[i].dyn_cast_or_null<DictionaryAttr>()) {
        NamedAttrList originAttrs = func.getResultAttrs(i);
        originAttrs.append(newResultAttrsDict.getValue());
        func.setResultAttrs(i, originAttrs.getDictionary(func->getContext()));
      }
    }
    func->removeAttr(resultAttrsName);
  }
}

void ConvertToByrePass::runOnOperation() {
  auto m = getOperation();
  OpPassManager pm(m.getOperationName());

  pm.addPass(createConvertFuncAndCallToByrePass(appendArgTypes));
  pm.addNestedPass<func::FuncOp>(createConvertLmhloToByrePass(appendArgTypes));

  if (mlir::failed(runPipeline(pm, m))) {
    signalPassFailure();
  }
}

void ConvertFuncAndCallToByrePass::runOnOperation() {
  ModuleOp m = getOperation();
  MLIRContext &ctx = getContext();
  llvm::SmallVector<func::FuncOp, 4> entryCollector;
  llvm::SmallVector<func::CallOp, 16> callCollector;
  llvm::SmallVector<func::FuncOp, 16> removeFuncCollector;

  identifyEntryPointFuncAndCalls(m, entryCollector, callCollector,
                                 removeFuncCollector);

  // early termination if module has no entry point function
  if (entryCollector.size() == 0) {
    return;
  }

  // insert byre.container_module to module if there is none.
  if (!m->hasAttr(byre::ByreDialect::getContainerModuleAttrName())) {
    m->setAttr(byre::ByreDialect::getContainerModuleAttrName(),
               UnitAttr::get(&ctx));
  }

  // rewrite private calls
  rewriteCallOpsForFuncOp(callCollector);

  unsigned inputCnt = 0, outputCnt = 0;
  for (auto func : entryCollector) {
    // Note: In this process we will give all arguments and results of given
    // func a unique `argName`, all arguments would be treated as argType::Input
    // and all results would be treated as argType::Output. But if argument of
    // func was with attribute placholders `argName` and `argType`, it will
    // overwrite those two attributes later.
    markFuncOpInOutTypeForLmhlo(func, inputCnt, outputCnt);

    rewriteByreResultAttrsToFuncResultAttr(func);

    relocateFuncOpResults(func, this->removeDupOutputs);

    removeAttrPlaceholders(func, attrNames);

    removeArgAttrPlaceholders(func, argAttrNames);
  }

  // Below rewrite std.call to byre.compute
  ConversionTarget target(getContext());
  target.addLegalDialect<byre::ByreDialect, func::FuncDialect,
                         memref::MemRefDialect, scf::SCFDialect,
                         ace::AceDialect>();

  target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

  target.addDynamicallyLegalOp<func::CallOp>([&](Operation *op) {
    auto func = op->getParentOfType<func::FuncOp>();
    return !isEntryPointFunc(func);
  });

  RewritePatternSet patterns(&ctx);
  populateStdToByreConversionPatterns(patterns, appendArgTypes);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(m, target, frozenPatterns))) {
    return signalPassFailure();
  }

  for (auto func : removeFuncCollector) {
    func->erase();
  }
}

void ConvertLmhloToByrePass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext &ctx = getContext();
  if (!isEntryPointFunc(func) && !isFuncWithEntryPointPlaceholder(func)) {
    return;
  }

  // Below rewrite lace ops, view Op
  {
    ConversionTarget target(getContext());
    target.addLegalDialect<byre::ByreDialect, memref::MemRefDialect>();
    target.addDynamicallyLegalDialect<lace::LaceDialect>([](Operation *op) {
      return !llvm::isa<lace::AliasLikeOpInterface>(op);
    });
    target.addIllegalOp<memref::ViewOp, memref::CopyOp>();
    RewritePatternSet patterns(&ctx);
    populateViewLikeToByreConversionPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }

  // Below rewrite Lmhlo ops
  {
    ConversionTarget target(getContext());
    target.addLegalDialect<byre::ByreDialect, func::FuncDialect,
                           memref::MemRefDialect, scf::SCFDialect>();

    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();

    target.addIllegalDialect<LmhloDialect, lace::LaceDialect>();

    RewritePatternSet patterns(&ctx);
    populateLmhloToByreConversionPatterns(patterns, lmhloSupportMap,
                                          appendArgTypes);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }

  // TODO move this to fold
  // remove unused fill op
  func.walk([&](byre::ComputeOp op) {
    if (op.getCallee() == "FillOp") {
      auto value = op->getOperand(0);
      if (value.hasOneUse() && value.getDefiningOp<memref::AllocOp>()) {
        op->erase();
      }
    }
  });
}

} // namespace

void mlir::populateLmhloToByreConversionPatterns(
    RewritePatternSet &patterns, llvm::StringMap<StringRef> &supportMap,
    bool appendArgTypes) {
  // clang-format off
  // TODO move this from a file
  // TODO use MACRO trick to add patterns
  patterns.add<ConvertToByrePattern<lmhlo::AddOp>,
               ConvertToByrePattern<lmhlo::ConvertOp>, 
               ConvertToByrePattern<lmhlo::GatherOp>,
               ConvertToByrePattern<lmhlo::ReshapeOp>,
               ConvertToByrePattern<lmhlo::ScatterOp>,
               ConvertToByrePattern<lmhlo::SliceOp>, 
               ConvertToByrePatternWithAllAttrs<lmhlo::TransposeOp>>(
                 patterns.getContext(),
                 supportMap, 
                 appendArgTypes);

  patterns.add<ConvertConstOpToByrePattern<lmhlo::ConstantOp>,
               ConvertConstOpToByrePattern<lace::ConstOp>,
               ConvertCustomCallOpToByrePattern<lmhlo::CustomCallOp>,
               ConvertCustomCallOpToByrePattern<lace::CustomCallOp>,
               ConvertDotOpToByrePattern,
               ConvertConvOpToByrePattern,
               ConvertReduceOpToByrePattern,
               ConvertReduceWindowOpToByrePattern, 
               ConvertSelectAndScatterOpToByrePattern>(
      patterns.getContext(), appendArgTypes);
  // clang-format on
}

void mlir::populateViewLikeToByreConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertAliasLikeOpToByrePattern,
               ConvertMemrefCopyOpToByrePattern, 
               ConvertViewOpToByrePattern>(
      patterns.getContext());
  // clang-format on
}

void mlir::populateStdToByreConversionPatterns(RewritePatternSet &patterns,
                                               bool appendArgTypes) {
  patterns.add<ConvertCallOpToByrePattern>(patterns.getContext(),
                                           appendArgTypes);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertToByrePass(bool appendArgTypes) {
  return std::make_unique<ConvertToByrePass>(appendArgTypes);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncAndCallToByrePass(bool appendArgTypes,
                                         bool removeDupOutputs) {
  return std::make_unique<ConvertFuncAndCallToByrePass>(appendArgTypes,
                                                        removeDupOutputs);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertLmhloToByrePass(bool appendArgTypes) {
  return std::make_unique<ConvertLmhloToByrePass>(appendArgTypes);
}
