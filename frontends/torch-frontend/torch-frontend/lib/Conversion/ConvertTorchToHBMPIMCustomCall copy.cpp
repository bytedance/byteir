//===- ConvertTorchToHBMPIMCustomCall.cpp ---------------------------*--- C++
//-*-===//
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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-frontend/Conversion/ConvertTorchToHBMPIMCustomCall.h"
#include "torch-frontend/Utils/CustomCallUtil.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/CustomOpUtils.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "utils/convert_op_folder.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
llvm::SmallVector<NamedAttribute> getDefaultAttrs(PatternRewriter &rewriter) {
  llvm::SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getStringAttr("has_side_effect"),
                     rewriter.getBoolAttr(false));
  attrs.emplace_back(rewriter.getStringAttr("backend_config"),
                     rewriter.getStringAttr(""));
  attrs.emplace_back(rewriter.getStringAttr("api_version"),
                     rewriter.getI32IntegerAttr(static_cast<int>(
                         mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL)));
  attrs.emplace_back(rewriter.getStringAttr("called_computations"),
                     rewriter.getArrayAttr({}));

  return attrs;
}


Value promoteType(Location loc, Value input, TensorType desiredType,
                  PatternRewriter &rewriter) {
  TensorType inType = input.getType().dyn_cast<TensorType>();
  if (inType.getElementType() == desiredType.getElementType()) {
    return input;
  }

  TensorType promotedType =
      inType.cloneWith(inType.getShape(), desiredType.getElementType());
  return rewriter.create<mhlo::ConvertOp>(loc, promotedType, input);
}
} // namespace

// convert addInt to custom

namespace {
class ConvertAtenAddOp : public OpConversionPattern<AtenAddOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();
    Value a = operands[0];
    Value b = operands[1];
    auto aType = a.getType().template cast<RankedTensorType>();
    auto bType = b.getType().template cast<RankedTensorType>();
    SmallVector<Value> bufferArgs({a, b});
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }
    std::vector<NamedAttribute> byteir_attrs;
    // byteir_attrs.emplace_back(rewriter.getStringAttr("broadcast_dims"),
    //                           rewriter.getI64ArrayAttr({}));
    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getAddHBMPIMName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));
    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    customCallOp->setAttr("device", rewriter.getStringAttr("hbmpim"));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

} // namespace

// // torch.aten.add.Int
namespace {
class ConvertAtenAddIntOp : public OpConversionPattern<AtenAddIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAddIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();
    Value a = operands[0];
    Value b = operands[1];
    auto aType = a.getType().template cast<RankedTensorType>();
    // auto bType = b.getType().template cast<RankedTensorType>();
    SmallVector<Value> bufferArgs({a, b});
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }
    std::vector<NamedAttribute> byteir_attrs;

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getAddScalarHBMPIMName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    customCallOp->setAttr("device", rewriter.getStringAttr("hbmpim"));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

} // namespace

// //torch.aten.add.float
namespace {
class ConvertAtenAddScalarOp : public OpConversionPattern<AtenAddScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAddScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();
    Value a = operands[0];
    Value b = operands[1];
    auto aType = a.getType().template cast<RankedTensorType>();

    // auto bType = b.getType().template cast<RankedTensorType>();
    SmallVector<Value> bufferArgs({a, b});
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }
    std::vector<NamedAttribute> byteir_attrs;
    // byteir_attrs.emplace_back(rewriter.getStringAttr("broadcast_dims"),
    //                           rewriter.getI64ArrayAttr({}));
    auto attrs =
        getDefaultAttrs(rewriter); // TODO: add float add scalar custom op
    if ((a.getType().cast<ShapedType>().getElementType().isa<Float16Type>())) {
      attrs.emplace_back(
          rewriter.getStringAttr("call_target_name"),
          rewriter.getStringAttr(getAddScalarHBMPIMName() + ".fp16"));
    } else if ((a.getType()
                    .cast<ShapedType>()
                    .getElementType()
                    .isa<Float32Type>())) {

      attrs.emplace_back(
          rewriter.getStringAttr("call_target_name"),
          rewriter.getStringAttr(getAddScalarHBMPIMName() + ".fp32"));
    } else {
      attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                         rewriter.getStringAttr(getAddScalarHBMPIMName()));
    }

    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));
    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});

    // if type is  float

    customCallOp->setAttr("device", rewriter.getStringAttr("hbmpim"));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

} // namespace

// torch.aten.add.Tensor
namespace {
class ConvertAtenAddTensorOp : public OpConversionPattern<AtenAddTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAddTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();
    Value a = operands[0];
    Value b = operands[1];
    auto aType = a.getType().template cast<RankedTensorType>();
    // auto bType = b.getType().template cast<RankedTensorType>();
    SmallVector<Value> bufferArgs({a, b});
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }
    std::vector<NamedAttribute> byteir_attrs;

    byteir_attrs.emplace_back(rewriter.getStringAttr("device"),
                              rewriter.getStringAttr("hbmpim"));
    auto attrs = getDefaultAttrs(rewriter);

    // check if a is float
    if ((a.getType().cast<ShapedType>().getElementType().isa<Float16Type>())) {
      attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                         rewriter.getStringAttr(getAddHBMPIMName() + ".fp16"));
    } else if ((a.getType()
                    .cast<ShapedType>()
                    .getElementType()
                    .isa<Float32Type>())) {

      attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                         rewriter.getStringAttr(getAddHBMPIMName() + ".fp32"));
    } else {
      attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                         rewriter.getStringAttr(getAddHBMPIMName()));
    }

    //         if (a.getType()
    //              .cast<ShapedType>()
    //              .getElementType()
    //              .isa<FloatType>()) {
    //    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
    //                      rewriter.getStringAttr(getAddScalarHBMPIMName()+".float"));
    //   } else {
    //  attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
    //                      rewriter.getStringAttr(getAddScalarHBMPIMName()));
    //   }
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));
    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    customCallOp->setAttr("device", rewriter.getStringAttr("hbmpim"));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

} // namespace

//convert mul to custom call
namespace {
class ConvertAtenMulTensorOp : public OpConversionPattern<AtenMulTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMulTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto operands = adaptor.getOperands();
    Value a = operands[0];
    Value b = operands[1];
    auto aType = a.getType().template cast<RankedTensorType>();
    auto bType = b.getType().template cast<RankedTensorType>();
    SmallVector<Value> bufferArgs({a, b});
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }
    std::vector<NamedAttribute> byteir_attrs;
    // byteir_attrs.emplace_back(rewriter.getStringAttr("broadcast_dims"),
    //                           rewriter.getI64ArrayAttr({}));
    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getMulHBMPIMName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));
    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    customCallOp->setAttr("device", rewriter.getStringAttr("hbmpim"));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

} // namespace


// matmul to custom call if gemv
// namespace {
// class ConvertAtenMatmulOp : public OpConversionPattern<AtenMatmulOp> {
// public:
//   using OpConversionPattern::OpConversionPattern;
//   LogicalResult
//   matchAndRewrite(AtenMatmulOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Value lhs = adaptor.getSelf();
//     Value rhs = adaptor.getOther();
//     auto lhsType = lhs.getType().template cast<RankedTensorType>();
//     auto rhsType = rhs.getType().template cast<RankedTensorType>();
//     SmallVector<Value> bufferArgs({lhs, rhs});

//     SmallVector<Type> resultTypes;
//     if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
//                                                 resultTypes))) {
//       return op.emitError("could not convert output types");
//     }
//     // check if gemv
//     if (lhsType.getRank() == 2 && rhsType.getRank() == 1) {
//       std::vector<NamedAttribute> byteir_attrs;
//       byteir_attrs.emplace_back(rewriter.getStringAttr("transpose_lhs"),
//                                 rewriter.getBoolAttr(false));
//       byteir_attrs.emplace_back(rewriter.getStringAttr("transpose_rhs"),
//                                 rewriter.getBoolAttr(false));

//       auto attrs = getDefaultAttrs(rewriter);
//       attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
//                          rewriter.getStringAttr(getGemvHBMPIMName()));
//       attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
//                          rewriter.getDictionaryAttr(byteir_attrs));

//       auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
//           op->getLoc(), resultTypes, bufferArgs,
//           ArrayRef<NamedAttribute>{attrs});
//       rewriter.replaceOp(op, customCallOp->getResults());
//       return success();

//     } else if (lhsType.getRank() == 1 && rhsType.getRank() == 2) {
//       std::vector<NamedAttribute> byteir_attrs;
//       byteir_attrs.emplace_back(rewriter.getStringAttr("transpose_lhs"),
//                                 rewriter.getBoolAttr(false));
//       byteir_attrs.emplace_back(rewriter.getStringAttr("transpose_rhs"),
//                                 rewriter.getBoolAttr(false));

//       auto attrs = getDefaultAttrs(rewriter);
//       attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
//                          rewriter.getStringAttr(getGemvHBMPIMName()));
//       attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
//                          rewriter.getDictionaryAttr(byteir_attrs));

//       auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
//           op->getLoc(), resultTypes, bufferArgs,
//           ArrayRef<NamedAttribute>{attrs});
//       rewriter.replaceOp(op, customCallOp->getResults());
//       return success();
//     }

//     else {
//       return rewriter.notifyMatchFailure(op, "unimplemented: "
//                                              "matmul is not gemv");
//     }

//     return success();
//   }
// };
// } // namespace

//

namespace {
class ConvertTorchToHBMPIMCustomCall
    : public ConvertTorchToHBMPIMCustomCallBase<
          ConvertTorchToHBMPIMCustomCall> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<stablehlo::StablehloDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, mhlo::MhloDialect,
                           arith::ArithDialect, tensor::TensorDialect,
                           stablehlo::StablehloDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    // target.addIllegalOp<AtenMatmulOp>();
    // patterns.add<ConvertAtenMatmulOp>(typeConverter, context);

    // target.addIllegalOp<AtenAddOp>();
    // patterns.add<ConvertAtenAddOp>(typeConverter, context);

    target.addIllegalOp<AtenAddTensorOp>();
    patterns.add<ConvertAtenAddTensorOp>(typeConverter, context);
    target.addIllegalOp<AtenAddIntOp>();
    patterns.add<ConvertAtenAddIntOp>(typeConverter, context);
    // target.addIllegalOp<AtenAddScalarOp>();
    // patterns.add<ConvertAtenAddScalarOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertTorchToHBMPIMCustomCall() {
  return std::make_unique<ConvertTorchToHBMPIMCustomCall>();
}
