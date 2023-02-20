//===- HloFusionToLinalg.cpp ----------------------------------*--- C++ -*-===//
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
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::mhlo;

namespace {

/// Code below is copied from legalize_to_linalg.cc
/// Remove this when upstream supports general float type reduce pattern

bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
  return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
}

bool verifyHloOpBufferOrTensorSemantics(Operation *op) {
  auto verifyType = [&](Value val) -> bool {
    return val.getType().isa<RankedTensorType>();
  };
  if (!llvm::all_of(op->getOperands(), verifyType))
    return false;
  return llvm::all_of(op->getResults(), verifyType);
}

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
      for (auto &en : llvm::enumerate(resultType.getShape())) {
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

class SoftmaxCustomCallConverter
    : public OpConversionPattern<mhlo::CustomCallOp> {
public:
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

class DotGeneralLinalgExtBatchMatMulOpConversion
    : public OpConversionPattern<mhlo::DotGeneralOp> {
public:
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

    mhlo::populateHloToLinalgConversionPattern(&ctx, *typeConverter, &patterns,
                                               enablePrimitiveOps);
    patterns.add<ReduceWindowOpConversion>(*typeConverter, &ctx,
                                           PatternBenefit(2));
    patterns.add<DotGeneralLinalgExtBatchMatMulOpConversion>(
        *typeConverter, &ctx, PatternBenefit(2));

    populateHloToLinalgExtConversionPattern(&ctx, patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(func, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateHloToLinalgExtConversionPattern(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<SoftmaxCustomCallConverter>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloFusionToLinalgPass(llvm::StringRef anchorTag,
                                  bool enablePrimitiveOps) {
  return std::make_unique<HloFusionToLinalgPass>(anchorTag, enablePrimitiveOps);
}
