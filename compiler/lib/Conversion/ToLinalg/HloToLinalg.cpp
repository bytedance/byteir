//===- HloToLinalg.cpp ----------------------------------------*--- C++ -*-===//
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
// Some code comes from legalize_to_linalg.cc in TensorFlow
// Original license:
//
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
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
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/legalize_to_linalg_utils.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::mhlo;

namespace {

/// Code below is copied from legalize_to_linalg.cc
/// Remove this when upstream supports general float type reduce pattern

bool verifyHloOpBufferOrTensorSemantics(Operation *op) {
  auto verifyType = [&](Value val) -> bool {
    return val.getType().isa<RankedTensorType>();
  };
  if (!llvm::all_of(op->getOperands(), verifyType))
    return false;
  return llvm::all_of(op->getResults(), verifyType);
}

// Util
Value fillTensorWithZeros(OpBuilder &builder, Location loc, Value tensor) {
  auto type = tensor.getType().cast<ShapedType>();
  Value zero;
  // Complex numbers are a special case.
  if (auto complexType = type.getElementType().dyn_cast<ComplexType>()) {
    auto zeroElement = builder.getZeroAttr(complexType.getElementType());
    auto zeroAttr = builder.getArrayAttr({zeroElement, zeroElement});
    zero = builder.create<complex::ConstantOp>(loc, complexType, zeroAttr);
  } else {
    auto zeroAttr = builder.getZeroAttr(type.getElementType());
    zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
  }
  return builder.create<linalg::FillOp>(loc, zero, tensor).result();
}

inline Value mapMhloOpToScalarOp(Operation *op, Type resultType,
                                 ValueRange operands, OpBuilder &builder) {
#define CASE(MHLO_OP)                                                          \
  .Case<MHLO_OP>([&](MHLO_OP mhloOp) {                                         \
    return mhlo::MhloOpToStdScalarOp::mapOp(mhloOp, resultType, operands,      \
                                            &builder);                         \
  })
  // clang-format off
  return llvm::TypeSwitch<Operation *, Value>(op)
    CASE(mhlo::AbsOp)
    CASE(mhlo::AddOp)
    CASE(mhlo::AndOp)
    CASE(mhlo::Atan2Op)
    CASE(mhlo::BitcastConvertOp)
    CASE(mhlo::CbrtOp)
    CASE(mhlo::CeilOp)
    CASE(mhlo::ClampOp)
    CASE(mhlo::ClzOp)
    CASE(mhlo::CompareOp)
    CASE(mhlo::ComplexOp)
    CASE(mhlo::ConvertOp)
    CASE(mhlo::CopyOp)
    CASE(mhlo::CosineOp)
    CASE(mhlo::DivOp)
    CASE(mhlo::ExpOp)
    CASE(mhlo::Expm1Op)
    CASE(mhlo::FloorOp)
    CASE(mhlo::ImagOp)
    CASE(mhlo::IsFiniteOp)
    CASE(mhlo::Log1pOp)
    CASE(mhlo::LogOp)
    CASE(mhlo::LogisticOp)
    CASE(mhlo::MaxOp)
    CASE(mhlo::MinOp)
    CASE(mhlo::MulOp)
    CASE(mhlo::NegOp)
    CASE(mhlo::NotOp)
    CASE(mhlo::OrOp)
    CASE(mhlo::PopulationCountOp)
    CASE(mhlo::PowOp)
    CASE(mhlo::RealOp)
    CASE(mhlo::ReducePrecisionOp)
    CASE(mhlo::RemOp)
    CASE(mhlo::RoundNearestEvenOp)
    CASE(mhlo::RoundOp)
    CASE(mhlo::RsqrtOp)
    CASE(mhlo::SelectOp)
    CASE(mhlo::ShiftLeftOp)
    CASE(mhlo::ShiftRightArithmeticOp)
    CASE(mhlo::ShiftRightLogicalOp)
    CASE(mhlo::SignOp)
    CASE(mhlo::SineOp)
    CASE(mhlo::SqrtOp)
    CASE(mhlo::SubtractOp)
    CASE(mhlo::TanhOp)
    CASE(mhlo::XorOp)
  .Default([](Operation *) { return Value(); });
  // clang-format on
#undef CASE
}

inline LogicalResult
remappingRegionFromMhloToLinalgExt(ConversionPatternRewriter &rewriter,
                                   Region &oldRegion, Region &newRegion) {
  Block *oldBlock = &oldRegion.front();
  SmallVector<Type> newBlockArgTypes =
      llvm::to_vector(llvm::map_range(oldBlock->getArgumentTypes(), [](Type t) {
        return t.cast<ShapedType>().getElementType();
      }));
  SmallVector<Location> newBlockArgLocs = llvm::to_vector(llvm::map_range(
      oldBlock->getArguments(), [](Value v) { return v.getLoc(); }));

  Block *newBlock =
      rewriter.createBlock(&newRegion, {}, newBlockArgTypes, newBlockArgLocs);

  IRMapping bvm;
  for (auto &&[oldArg, newArg] :
       llvm::zip(oldBlock->getArguments(), newBlock->getArguments())) {
    bvm.map(oldArg, newArg);
  }
  for (auto &&op : oldBlock->without_terminator()) {
    SmallVector<Value> newOperands = llvm::to_vector(llvm::map_range(
        op.getOperands(), [&](Value v) { return bvm.lookup(v); }));
    Value newValue = mapMhloOpToScalarOp(
        &op, op.getResult(0).getType().cast<ShapedType>().getElementType(),
        newOperands, rewriter);
    if (!newValue) {
      return failure();
    }
    bvm.map(op.getResult(0), newValue);
  }
  Operation *oldTerminator = oldBlock->getTerminator();
  rewriter.create<linalg_ext::YieldOp>(
      oldTerminator->getLoc(),
      llvm::to_vector(llvm::map_range(oldTerminator->getOperands(),
                                      [&](Value v) { return bvm.lookup(v); })));
  return success();
}

struct ReduceWindowOpConversion
    : public OpConversionPattern<mhlo::ReduceWindowOp> {
  using OpConversionPattern<mhlo::ReduceWindowOp>::OpConversionPattern;

  /// mhlo.reduce_window is mapped to a linalg.pooling operation. The type of
  /// the pooling is determined based on the body of the reduce window
  /// operation. This class enumerates the different variants.
  enum class PoolingType {
    kInvalid,
    k2DMin,
    k3DMin,
    k2DMax,
    k3DMax,
    k2DAdd,
    k3DAdd,
  };

  static PoolingType getPoolingType(mhlo::ReduceWindowOp reduceOp,
                                    int resultIndex) {
    auto rank =
        reduceOp.getResultTypes()[resultIndex].cast<ShapedType>().getRank();
    if (Operation *op = reduceOp.getReductionOp(resultIndex)) {
      if (isa<mhlo::MinOp>(*op) && rank == 4)
        return PoolingType::k2DMin;
      if (isa<mhlo::MinOp>(*op) && rank == 5)
        return PoolingType::k3DMin;
      if (isa<mhlo::MaxOp>(*op) && rank == 4)
        return PoolingType::k2DMax;
      if (isa<mhlo::MaxOp>(*op) && rank == 5)
        return PoolingType::k3DMax;
      if (isa<mhlo::AddOp>(*op) && rank == 4)
        return PoolingType::k2DAdd;
      if (isa<mhlo::AddOp>(*op) && rank == 5)
        return PoolingType::k3DAdd;
    }
    return PoolingType::kInvalid;
  }

  LogicalResult
  matchAndRewrite(mhlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    int rank = op.getResultTypes()[0].cast<ShapedType>().getRank();
    if (rank != 4 && rank != 5) {
      return rewriter.notifyMatchFailure(
          op, "expected NHWC/NDHWC pooling-based op");
    }

    if (op.getPadding() && !isSplatValue(*op.getPadding(), 0)) {
      return rewriter.notifyMatchFailure(op, "require paddings are all zero");
    }

    if (op.getBaseDilations() && !isSplatValue(*op.getBaseDilations(), 1)) {
      return rewriter.notifyMatchFailure(op, "expected undilated base");
    }

    int lastDim = rank - 1;
    SmallVector<int64_t, 2> fakeWindowShapes;
    for (int i = 1; i < lastDim; ++i) {
      fakeWindowShapes.push_back(
          op.getWindowDimensions().getValues<int64_t>()[i]);
    }

    if (op.getWindowStrides() &&
        (op.getWindowStrides().value().getValues<int64_t>()[0] != 1 ||
         op.getWindowStrides().value().getValues<int64_t>()[lastDim] != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_strides to be [1,x,y,(z),1]");
    }
    if (op.getWindowDimensions() &&
        (op.getWindowDimensions().getValues<int64_t>()[0] != 1 ||
         op.getWindowDimensions().getValues<int64_t>()[lastDim] != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_dimensions to be [1,x,y,(z),1]");
    }

    Attribute strides;
    SmallVector<int64_t> vec;
    if (op.getWindowStridesAttr()) {
      for (int i = 1; i < lastDim; ++i) {
        vec.push_back(op.getWindowStrides().value().getValues<int64_t>()[i]);
      }
    } else {
      vec.assign(rank - 2, 1);
    }
    strides = rewriter.getI64VectorAttr(vec);

    Attribute dilations;
    vec.clear();
    if (op.getWindowDilations()) {
      for (int i = 1; i < lastDim; ++i) {
        vec.push_back(op.getWindowDilations().value().getValues<int64_t>()[i]);
      }
    } else {
      vec.assign(rank - 2, 1);
    }
    dilations = rewriter.getI64VectorAttr(vec);

    SmallVector<Value> poolingOps;

    ValueRange operands = adaptor.getInputs();
    ValueRange initValues = adaptor.getInitValues();
    for (auto it : llvm::zip(op.getResults(), operands, initValues)) {
      OpResult result = std::get<0>(it);
      Value input = std::get<1>(it);
      Value initValue = std::get<2>(it);
      auto resultType = result.getType().cast<ShapedType>();
      if (!input.getType()
               .cast<ShapedType>()
               .getElementType()
               .isa<FloatType>()) {
        return rewriter.notifyMatchFailure(op,
                                           "expected element type to be float");
      }

      // Create a fake window dimension.
      auto fakeWindowDims = rewriter.create<tensor::EmptyOp>(
          loc, fakeWindowShapes, resultType.getElementType());

      SmallVector<Value> resultDynamicDims;
      for (const auto &en : llvm::enumerate(resultType.getShape())) {
        if (en.value() != ShapedType::kDynamic)
          continue;
        Value dimSize = rewriter.create<tensor::DimOp>(loc, input, en.index());
        if (en.index() == 0 || static_cast<int64_t>(en.index()) == rank - 1) {
          // batch dims and channel dims can be derived from input dims
          // directly.
          resultDynamicDims.push_back(dimSize);
        } else {
          auto i = en.index() - 1;
          auto stride =
              strides.cast<DenseIntElementsAttr>().getValues<int64_t>()[i];
          auto dilation =
              dilations.cast<DenseIntElementsAttr>().getValues<int64_t>()[i];
          // let j = i * stride
          // output[i] = reduce( input[j, j + window_size * dilation) )
          Value offset = rewriter.create<arith::ConstantIndexOp>(
              loc, fakeWindowShapes[i] * dilation);
          dimSize = rewriter.create<arith::SubIOp>(loc, dimSize, offset);
          dimSize = rewriter.create<arith::DivUIOp>(
              loc, dimSize,
              rewriter.create<arith::ConstantIndexOp>(loc, stride));
          dimSize = rewriter.create<arith::AddIOp>(
              loc, dimSize, rewriter.create<arith::ConstantIndexOp>(loc, 1));
          resultDynamicDims.push_back(dimSize);
        }
      }
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), resultType.getElementType(),
          resultDynamicDims);

      initValue = rewriter.create<tensor::ExtractOp>(loc, initValue);
      Value filledInitTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, emptyTensor)
              .getResult(0);
      auto createOp = [&](auto *typePtr) -> linalg::LinalgOp {
        return cast<linalg::LinalgOp>(
            rewriter
                .create<std::remove_pointer_t<decltype(typePtr)>>(
                    loc, ArrayRef<Type>{resultType},
                    ValueRange{input, fakeWindowDims.getResult()},
                    filledInitTensor, strides, dilations,
                    linalg::getPrunedAttributeList(op))
                .getOperation());
      };
      linalg::LinalgOp poolingOp;
      PoolingType poolingType = getPoolingType(op, result.getResultNumber());
      switch (poolingType) {
      case PoolingType::k2DMin: {
        poolingOp = createOp(static_cast<linalg::PoolingNhwcMinOp *>(nullptr));
        break;
      }
      case PoolingType::k3DMin: {
        poolingOp = createOp(static_cast<linalg::PoolingNdhwcMinOp *>(nullptr));
        break;
      }
      case PoolingType::k2DMax: {
        poolingOp = createOp(static_cast<linalg::PoolingNhwcMaxOp *>(nullptr));
        break;
      }
      case PoolingType::k3DMax: {
        poolingOp = createOp(static_cast<linalg::PoolingNdhwcMaxOp *>(nullptr));
        break;
      }
      case PoolingType::k2DAdd: {
        poolingOp = createOp(static_cast<linalg::PoolingNhwcSumOp *>(nullptr));
        break;
      }
      case PoolingType::k3DAdd: {
        poolingOp = createOp(static_cast<linalg::PoolingNdhwcSumOp *>(nullptr));
        break;
      }
      case PoolingType::kInvalid:
        return rewriter.notifyMatchFailure(op, "unknown reduction operation");
      }
      poolingOps.push_back(poolingOp->getResult(0));
    }
    rewriter.replaceOp(op, poolingOps);
    return success();
  }
};

struct SoftmaxCustomCallConverter
    : public OpConversionPattern<mhlo::CustomCallOp> {
  using OpConversionPattern<mhlo::CustomCallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp op, mhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != getSoftmaxName())
      return failure();

    auto attr = op->getAttrOfType<DictionaryAttr>(getCustomCallAttrName());
    auto axisAttr = attr.getAs<IntegerAttr>("axis");
    assert(axisAttr && "Softmax custom call axis attribute not found.");

    int axis = axisAttr.getInt();
    rewriter.setInsertionPoint(op);
    auto resultType =
        op->getResult(0).getType().dyn_cast_or_null<RankedTensorType>();
    assert(resultType && "Dynamic shape not supported yet.");

    SmallVector<int64_t> sizes;
    sizes.reserve(resultType.getRank() - 1);
    for (int i = 0; i < resultType.getRank(); i++) {
      if (i != axis)
        sizes.push_back(resultType.getShape()[i]);
    }

    auto loc = op->getLoc();

    assert(resultType.getElementType().isIntOrFloat());
    Value zeroCst;
    Value negInf;
    if (IntegerType intType =
            resultType.getElementType().dyn_cast<IntegerType>()) {
      zeroCst = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0,
                                                            intType.getWidth());
      negInf = rewriter.create<mlir::arith::ConstantIntOp>(
          loc, APInt::getSignedMinValue(intType.getWidth()).getSExtValue(),
          intType.getWidth());
    }

    if (FloatType floatType =
            resultType.getElementType().dyn_cast<FloatType>()) {
      zeroCst = rewriter.create<mlir::arith::ConstantFloatOp>(
          loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
      negInf = rewriter.create<mlir::arith::ConstantFloatOp>(
          loc, APFloat::getInf(floatType.getFloatSemantics(), true), floatType);
    }

    auto output = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    auto max = rewriter.create<tensor::EmptyOp>(loc, sizes,
                                                resultType.getElementType());
    auto accum = rewriter.create<tensor::EmptyOp>(loc, sizes,
                                                  resultType.getElementType());
    auto scale = rewriter.create<tensor::EmptyOp>(loc, sizes,
                                                  resultType.getElementType());

    Value filledMax =
        rewriter.create<linalg::FillOp>(loc, negInf, max.getResult())
            .getResult(0);
    Value filledAccum =
        rewriter.create<linalg::FillOp>(loc, zeroCst, accum.getResult())
            .getResult(0);

    auto softmax = rewriter.create<linalg_ext::SoftmaxOp>(
        loc,
        TypeRange{output.getResult().getType(), max.getResult().getType(),
                  accum.getResult().getType(), scale.getResult().getType()},
        op->getOperands(), ValueRange{output, filledMax, filledAccum, scale},
        axisAttr);
    rewriter.replaceOp(op, softmax.getResult(0));
    return success();
  }
};

struct DotGeneralLinalgExtBatchMatMulOpConversion
    : public OpConversionPattern<mhlo::DotGeneralOp> {
  using OpConversionPattern<mhlo::DotGeneralOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mhlo::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!verifyHloOpBufferOrTensorSemantics(op)) {
      return failure();
    }
    int64_t rank = op.getType().cast<RankedTensorType>().getRank();
    if (rank < 4) {
      return rewriter.notifyMatchFailure(
          op, "expected a batch matmul of rank >= 4");
    }

    auto isSeqFromZero = [](ArrayRef<int64_t> seq) {
      int64_t cnt = 0;
      for (int64_t v : seq) {
        if (v != cnt)
          return false;
        cnt++;
      }
      return true;
    };

    mhlo::DotDimensionNumbersAttr dimNumbers = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dimNumbers.getRhsContractingDimensions();
    if (!isSeqFromZero(lhsBatchingDims) &&
        static_cast<int64_t>(lhsBatchingDims.size()) != rank - 2)
      return rewriter.notifyMatchFailure(
          op,
          "expected lhs batching dimensions as a sequence from zero: [0, 1, "
          "2, ..., rank-2]");
    if (!isSeqFromZero(rhsBatchingDims) &&
        static_cast<int64_t>(rhsBatchingDims.size()) != rank - 2)
      return rewriter.notifyMatchFailure(
          op,
          "expected lhs batching dimensions as a sequence from zero: [0, 1, "
          "2, ..., rank-2]");
    if (lhsContractingDims.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs contracting dimensions size of 1");
    }
    if (rhsContractingDims.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs contracting dimensions size of 1");
    }

    std::string layout = "nn";
    if (lhsContractingDims[0] == rank - 2)
      layout[0] = 't';
    if (rhsContractingDims[0] == rank - 1)
      layout[0] = 't';

    Location loc = op.getLoc();
    // Convert unsigned to signed. This works because signed and unsigned
    // integer matmul is the same operation in two's complement.
    auto outputType =
        typeConverter->convertType(op.getType()).cast<ShapedType>();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, outputType, op, adaptor.getOperands());
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);
    Operation *linalgOp = rewriter.create<linalg_ext::BatchMatmulOp>(
        loc, adaptor.getLhs(), adaptor.getRhs(), zeroTensor, layout,
        linalg::getPrunedAttributeList(op));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

class ScatterOpConversion : public OpConversionPattern<mhlo::ScatterOp> {
public:
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mhlo::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO: batched scattering is not supported now
    if (adaptor.getUpdates().size() != 1 || adaptor.getInputs().size() != 1)
      return failure();

    Value indices = adaptor.getScatterIndices(),
          update = adaptor.getUpdates()[0], src = adaptor.getInputs()[0];
    auto isRankedTensorType = [](Value v) {
      if (auto tensorType = v.getType().dyn_cast_or_null<TensorType>()) {
        if (tensorType.hasRank()) {
          return true;
        }
      }
      return false;
    };
    // all of operands should be ranked tensor
    if (!(isRankedTensorType(indices) && isRankedTensorType(update) &&
          isRankedTensorType(src)))
      return failure();

    Location loc = op.getLoc();
    ShapedType indicesType = indices.getType().cast<ShapedType>(),
               updateType = update.getType().cast<ShapedType>(),
               srcType = src.getType().cast<ShapedType>();

    mhlo::ScatterDimensionNumbersAttr scatterDimensionNumbers =
        adaptor.getScatterDimensionNumbers();

    int64_t indicesRank = indicesType.getRank(),
            updateRank = updateType.getRank(), srcRank = srcType.getRank(),
            indexVectorDim = scatterDimensionNumbers.getIndexVectorDim();
    bool implicitTrailingOne = indexVectorDim == indicesRank;
    if ((!implicitTrailingOne) && indicesType.isDynamicDim(indicesRank - 1)) {
      return failure();
    }
    int64_t indicesDepth =
        implicitTrailingOne ? 1 : indicesType.getDimSize(indicesRank - 1);
    ArrayRef<int64_t> updateWindowDims =
                          scatterDimensionNumbers.getUpdateWindowDims(),
                      insertedWindowDims =
                          scatterDimensionNumbers.getInsertedWindowDims(),
                      scatterDimsToOperandDims =
                          scatterDimensionNumbers.getScatterDimsToOperandDims();
    int64_t updateWindowRank = srcRank - insertedWindowDims.size();
    if (updateWindowRank + indicesRank - 1 + implicitTrailingOne != updateRank)
      return failure();

    // check scatter to first `indicesDepth` dimensinos
    if (!llvm::equal(scatterDimsToOperandDims,
                     llvm::seq<int64_t>(0, indicesDepth)))
      return failure();

    // check insert window to first `indicesDepth` dimensinos
    if (!llvm::equal(insertedWindowDims, llvm::seq<int64_t>(0, indicesDepth)))
      return failure();

    // window dimensions should be the last `updateWindowRank` dimensions of
    // update
    if (!llvm::equal(
            updateWindowDims,
            llvm::seq<int64_t>(updateRank - updateWindowRank, updateRank)))
      return failure();

    if (!llvm::equal(updateType.getShape().take_back(updateWindowRank),
                     srcType.getShape().take_back(updateWindowRank)))
      return failure();

    Value newIndices = indices;
    if (implicitTrailingOne) {
      // expand indices's shape with implicit trailing one
      SmallVector<int64_t> newShape(indicesType.getShape());
      newShape.push_back(1);
      ShapedType targetType = indicesType.clone(newShape);
      SmallVector<ReassociationIndices> reassociations =
          llvm::to_vector(llvm::map_range(
              llvm::seq<int64_t>(0, indicesRank),
              [](int64_t index) { return ReassociationIndices{index}; }));
      reassociations.back().push_back(indicesRank);
      newIndices = rewriter.create<tensor::ExpandShapeOp>(
          loc, targetType, indices, reassociations);
    }

    // Copy src to newly allocated src for out-of-place scattering
    // TODO: copy if src RAW happens or src is a constant, otherwise
    // scattering could be applied inplace
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < srcType.getRank(); ++i) {
      if (srcType.isDynamicDim(i)) {
        dynamicSizes.push_back(getDimValue(rewriter, loc, src, i));
      }
    }
    Value empty = rewriter.create<tensor::EmptyOp>(loc, srcType, dynamicSizes)
                      .getResult();
    Value newSrc =
        rewriter.create<linalg::CopyOp>(loc, src, empty)->getResult(0);

    // TODO: respect indices_are_sorted
    auto scatterOp = rewriter.create<linalg_ext::ScatterOp>(
        loc, TypeRange{newSrc.getType()},
        /* inputs */ ValueRange{newIndices, update},
        /* outputs */ ValueRange{newSrc});

    // convert mhlo region to linalg_ext region
    if (failed(remappingRegionFromMhloToLinalgExt(
            rewriter, op.getUpdateComputation(), scatterOp.getRegion())))
      return failure();

    rewriter.replaceOp(op, scatterOp->getResults());

    return success();
  }
};

class LayerNormCustomCallConverter
    : public OpConversionPattern<mhlo::CustomCallOp> {
public:
  using OpConversionPattern<mhlo::CustomCallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp op, mhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != getLayerNormName())
      return failure();

    auto attr = op->getAttrOfType<DictionaryAttr>(getCustomCallAttrName());
    auto axisAttr = attr.getAs<ArrayAttr>("axis");
    assert(axisAttr && "LayerNorm custom call axis attribute not found.");

    auto epsAttr = attr.getAs<FloatAttr>("epsilon");
    assert(epsAttr && "LayerNorm custom call epsilon attribute not found.");

    rewriter.setInsertionPoint(op);
    auto resultType =
        op->getResult(0).getType().dyn_cast_or_null<RankedTensorType>();
    assert(resultType && "Dynamic shape not supported yet.");
    assert(resultType.getElementType().isIntOrFloat());

    auto loc = op->getLoc();
    auto numResults = op->getNumResults();

    auto output = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    if (numResults > 1) {
      auto meanType =
          op->getResult(1).getType().dyn_cast_or_null<RankedTensorType>();
      assert(meanType && "Dynamic shape not supported yet.");
      auto rstdType =
          op->getResult(2).getType().dyn_cast_or_null<RankedTensorType>();
      assert(rstdType && "Dynamic shape not supported yet.");
      auto mean = rewriter.create<tensor::EmptyOp>(loc, meanType.getShape(),
                                                   resultType.getElementType());
      auto rstd = rewriter.create<tensor::EmptyOp>(loc, rstdType.getShape(),
                                                   resultType.getElementType());

      auto layerNorm = rewriter.create<linalg_ext::LayerNormOp>(
          loc,
          TypeRange{output.getResult().getType(), mean.getResult().getType(),
                    rstd.getResult().getType()},
          op->getOperands(), ValueRange{output, mean, rstd}, axisAttr, epsAttr);
      rewriter.replaceOp(op, layerNorm.getResult(0));
    } else {
      auto layerNorm = rewriter.create<linalg_ext::LayerNormOp>(
          loc, TypeRange{output.getResult().getType()}, op->getOperands(),
          ValueRange{output}, axisAttr, epsAttr);
      rewriter.replaceOp(op, layerNorm.getResult(0));
    }

    return success();
  }
};

class GeLUCustomCallConverter : public OpConversionPattern<mhlo::CustomCallOp> {
public:
  using OpConversionPattern<mhlo::CustomCallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp op, mhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != getGeLUName())
      return failure();

    auto attr = op->getAttrOfType<DictionaryAttr>(getCustomCallAttrName());
    if (!attr)
      return failure();

    auto approximateAttr = attr.getAs<StringAttr>("approximate");
    if (!approximateAttr)
      return failure();

    llvm::StringRef approximate = approximateAttr.getValue();
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder = nullptr;

    if (approximate == "erf") {
      bodyBuilder = [&](OpBuilder &builder, Location loc, ValueRange args) {
        // 0.5x * (1 + erf(x/sqrt(2)))
        auto x = args[0];
        auto elementType = x.getType();
        ImplicitLocOpBuilder b(loc, builder);
        Value sqrt1_2 = b.create<arith::ConstantOp>(
            b.getFloatAttr(elementType, 0.707106781f));
        Value erf_arg = b.create<arith::MulFOp>(x, sqrt1_2);
        Value erf = b.create<math::ErfOp>(erf_arg);
        Value constant_one =
            b.create<arith::ConstantOp>(b.getFloatAttr(elementType, 1.f));
        Value erf_plu_one = b.create<arith::AddFOp>(erf, constant_one);
        Value constant_half =
            b.create<arith::ConstantOp>(b.getFloatAttr(elementType, .5f));
        Value x_half = b.create<arith::MulFOp>(constant_half, x);
        Value result = b.create<arith::MulFOp>(x_half, erf_plu_one);
        b.create<linalg::YieldOp>(result);
      };
    } else if (approximate == "tanh") {
      bodyBuilder = [&](OpBuilder &builder, Location loc, ValueRange args) {
        // 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        auto x = args[0];
        auto elementType = x.getType();
        ImplicitLocOpBuilder b(loc, builder);
        Value x2 = b.create<arith::MulFOp>(x, x);
        Value x3 = b.create<arith::MulFOp>(x2, x);
        Value coeff0 =
            b.create<arith::ConstantOp>(b.getFloatAttr(elementType, 0.044715f));
        Value y0 = b.create<arith::MulFOp>(coeff0, x3);
        Value y1 = b.create<arith::AddFOp>(x, y0);
        Value coeff1 = b.create<arith::ConstantOp>(
            b.getFloatAttr(elementType, 0.79788456f)); // sqrt(2/pi)
        Value y2 = b.create<arith::MulFOp>(coeff1, y1);
        Value tanh = b.create<math::TanhOp>(y2);
        Value constant_one =
            b.create<arith::ConstantOp>(b.getFloatAttr(elementType, 1.f));
        Value tanh_plus_one = b.create<arith::AddFOp>(constant_one, tanh);
        Value constant_half =
            b.create<arith::ConstantOp>(b.getFloatAttr(elementType, .5f));
        Value x_half = b.create<arith::MulFOp>(constant_half, x);
        Value result = b.create<arith::MulFOp>(x_half, tanh_plus_one);
        b.create<linalg::YieldOp>(result);
      };
    }

    if (!bodyBuilder)
      return failure();

    auto loc = op.getLoc();
    auto input = adaptor.getInputs()[0];
    auto inputType = input.getType().cast<ShapedType>();
    auto rank = inputType.getRank();
    SmallVector<AffineMap> affineMaps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    Value emptyTensor =
        rewriter
            .create<tensor::EmptyOp>(
                loc,
                llvm::to_vector(llvm::map_range(llvm::seq<int64_t>(0, rank),
                                                [&](int64_t dim) {
                                                  return getDim(rewriter, loc,
                                                                input, dim);
                                                })),
                inputType.getElementType())
            .getResult();
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{emptyTensor.getType()}, ValueRange{adaptor.getOperands()},
        ValueRange{emptyTensor}, affineMaps, iteratorTypes, bodyBuilder,
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

static Value castToIndexTensor(OpBuilder &builder, Location loc,
                               Value shapeOp) {
  ShapedType resultTy = shape::getExtentTensorType(
      builder.getContext(), shapeOp.getType().cast<ShapedType>().getDimSize(0));
  if (shapeOp.getType() == resultTy)
    return shapeOp; // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, resultTy, shapeOp);
}

class RngUniformCustomCallConverter
    : public OpConversionPattern<mhlo::CustomCallOp> {
public:
  using OpConversionPattern<mhlo::CustomCallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp op, mhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != getRngUniformName())
      return failure();
    auto ctx = op.getContext();
    auto minTy = adaptor.getOperands()[0].getType().dyn_cast<ShapedType>();
    auto maxTy = adaptor.getOperands()[1].getType().dyn_cast<ShapedType>();
    if (!minTy.getElementType().dyn_cast<FloatType>() ||
        !maxTy.getElementType().dyn_cast<FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "expected min/max for rng op to be FloatType");
    }
    auto targetTy = op.getResults()[0].getType().cast<ShapedType>();
    if (!targetTy) {
      return rewriter.notifyMatchFailure(
          op, "expected target shape of rng op to be ShapedType");
    }
    auto loc = op.getLoc();

    // create empty tensor
    SmallVector<Value> sizes;
    if (adaptor.getOperands().size() == 5) {
      // dynamic shape
      auto reifiedShape =
          castToIndexTensor(rewriter, loc, adaptor.getOperands()[4]);
      for (const auto &en : llvm::enumerate(targetTy.getShape())) {
        if (en.value() != ShapedType::kDynamic)
          continue;
        sizes.push_back(rewriter.create<tensor::ExtractOp>(
            loc, reifiedShape,
            ValueRange{
                rewriter.create<arith::ConstantIndexOp>(loc, en.index())}));
      }
    } else {
      // static shape
      assert(adaptor.getOperands().size() == 4 &&
             "static shape byteir.rng_uniform must have 4 operands.");
    }
    Value emptyTensor = getEmptyTensor(rewriter, loc, targetTy, sizes);

    // Creates index map using target matrix's rank.
    auto targetRank = targetTy.getRank();
    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.push_back(AffineMap::get(targetRank, 0, ctx)); // low
    indexingMaps.push_back(AffineMap::get(targetRank, 0, ctx)); // high
    indexingMaps.push_back(AffineMap::get(targetRank, 0, ctx)); // seed
    indexingMaps.push_back(AffineMap::get(targetRank, 0, ctx)); // offset
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(targetRank, ctx));
    // Generic region with LCG Algorithm that make use of element index from:
    // https://reviews.llvm.org/D101364
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/targetTy,
        /*inputs=*/
        ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1],
                   adaptor.getOperands()[2], adaptor.getOperands()[3]},
        /*outputs=*/emptyTensor, indexingMaps,
        getParallelAndReductionIterators(/*nLoops=*/targetRank,
                                         /*nReduction=*/0),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value i32Seed =
              b.create<arith::TruncIOp>(loc, b.getI32Type(), args[2]);
          Value i32Offset =
              b.create<arith::TruncIOp>(loc, b.getI32Type(), args[3]);
          Value kInitialSeed = b.create<arith::AddIOp>(loc, i32Seed, i32Offset);
          llvm::SmallVector<Value> updateVec = {kInitialSeed}; // seed
          Value multiplier =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(1103515245));
          Value incrementStep =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(12345));
          // For output matrix with rank N:
          // temp1 = (cast(I32, index(D.0)) + seed) * mult + incr
          // ...
          // tempN = (cast(I32, index(D.(N))) + tempN_1) * mult + incr
          for (int i = 0; i < targetRank; i++) {
            Value update = updateVec.back();
            Value ind = b.create<linalg::IndexOp>(loc, i);
            Value castInd =
                b.create<arith::IndexCastOp>(loc, b.getI32Type(), ind);
            Value addRes = b.create<arith::AddIOp>(loc, castInd, update);
            Value multRes = b.create<arith::MulIOp>(loc, addRes, multiplier);
            Value incRes = b.create<arith::AddIOp>(loc, multRes, incrementStep);
            updateVec.push_back(incRes);
          }
          // Scaling = (max - min) * const(F64, 2.3283064E-10)
          // which is derived from rand(min,max) = rand()/(RAND_MAX/(max-min)).
          Value epsilon = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(args[0].getType(), 2.3283064E-10));
          Value range = b.create<arith::SubFOp>(loc, args[1], args[0]);
          Value scale = b.create<arith::MulFOp>(loc, range, epsilon);
          // Res = cast(T, cast(F64, tempN) * scaling + min)
          Value updateCast = b.create<arith::UIToFPOp>(
              loc, targetTy.getElementType(), updateVec.back());
          Value scaleUpdate = b.create<arith::MulFOp>(loc, updateCast, scale);
          Value res = b.create<arith::AddFOp>(loc, scaleUpdate, args[0]);
          b.create<linalg::YieldOp>(loc, res);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};
struct HloFusionToLinalgPass
    : public HloFusionToLinalgBase<HloFusionToLinalgPass> {

  HloFusionToLinalgPass(StringRef tag, bool enablePrimitiveOps)
      : HloFusionToLinalgBase() {
    anchorTag = tag.str();
    this->enablePrimitiveOps = enablePrimitiveOps;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, linalg_ext::LinalgExtDialect,
                    scf::SCFDialect, math::MathDialect, memref::MemRefDialect,
                    shape::ShapeDialect>();
  }

  void runOnOperation() final {
    func::FuncOp func = getOperation();

    bool valid = anchorTag.empty() || func->hasAttrOfType<UnitAttr>(anchorTag);

    // early termination
    if (!valid)
      return;

    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<
        arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
        linalg::LinalgDialect, math::MathDialect, tensor::TensorDialect,
        scf::SCFDialect, shape::ShapeDialect, linalg_ext::LinalgExtDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    auto typeConverter = createHloToLinalgTypeConverter();

    mhlo::populateScalarHloToArithmeticConversionPatterns(
        &ctx, *typeConverter, &patterns,
        [](Operation *op) { return isInBodyOfLinalgOps(op); });
    mhlo::populateHloToLinalgConversionPattern(&ctx, *typeConverter, &patterns,
                                               enablePrimitiveOps);
    populateHloToLinalgExtConversionPattern(*typeConverter, patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateHloToLinalgExtConversionPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  auto ctx = patterns.getContext();
  patterns.add<ReduceWindowOpConversion>(typeConverter, ctx, PatternBenefit(2));
  patterns.add<DotGeneralLinalgExtBatchMatMulOpConversion>(typeConverter, ctx,
                                                           PatternBenefit(2));
  patterns.add<SoftmaxCustomCallConverter>(ctx);
  patterns.add<ScatterOpConversion>(ctx);
  patterns.add<LayerNormCustomCallConverter>(ctx);
  patterns.add<GeLUCustomCallConverter>(ctx);
  patterns.add<RngUniformCustomCallConverter>(ctx);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloFusionToLinalgPass(llvm::StringRef anchorTag,
                                  bool enablePrimitiveOps) {
  return std::make_unique<HloFusionToLinalgPass>(anchorTag, enablePrimitiveOps);
}
