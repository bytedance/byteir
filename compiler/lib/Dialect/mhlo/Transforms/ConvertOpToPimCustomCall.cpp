//===- ConvertOpToPimCustomCall.cpp ------------------------------*--- C++
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

#include "byteir/Dialect/mhlo/Transforms/ConvertOpToPimCustomCall.h"
#include "./PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

func::FuncOp getOrCreatePrivateFunctionDeclare(ModuleOp module,
                                               const std::string &funcName,
                                               const std::string &byreOpName,
                                               FunctionType funcType) {
  auto func = SymbolTable(module).lookup<func::FuncOp>(funcName);
  if (func) {
    // TODO(lyq): check func's type == funcType, and check func's attr
    return func;
  } else {
    MLIRContext *context = module.getContext();
    OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
    func = builder.create<func::FuncOp>(UnknownLoc::get(context), funcName,
                                        funcType);
    func.setPrivate();
    func->setAttr(byre::getByreComputeName(),
                  builder.getStringAttr(byreOpName));
    func->setAttr(byre::getByreForceComputeNameAttrName(),
                  UnitAttr::get(context));
    return func;
  }
}

func::CallOp getOrCreateCallGetSeedOp(func::FuncOp func,
                                      func::FuncOp getSeedFunc,
                                      PatternRewriter &rewriter) {
  func::CallOp callGetSeedOp;
  func.walk([&](func::CallOp op) {
    if (getFuncOp(op) == getSeedFunc) {
      callGetSeedOp = op;
    }
  });
  if (!callGetSeedOp) {
    callGetSeedOp = rewriter.create<func::CallOp>(
        UnknownLoc::get(rewriter.getContext()), getSeedFunc, ArrayRef<Value>{});
  }
  // move func.call @getSeed to the begin of func
  Block *block = callGetSeedOp->getBlock();
  callGetSeedOp->moveBefore(&block->front());
  return callGetSeedOp;
}

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

// dotGeneral to gemv

struct ConvertDotGeneralToGEMVCustomOp
    : public OpRewritePattern<mhlo::DotGeneralOp> {
public:
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,

                                PatternRewriter &rewriter) const override {
    // check if dotGeneral can be converted to gemv

    if (op->getNumOperands() != 2) {
      return failure();
    }
    auto lhsType = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
    auto rhsType = op->getOperand(1).getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType) {
      return failure();
    }
    // if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
    //   return failure();
    // }
    // if (lhsType.getShape()[1] != rhsType.getShape()[0]) {
    //   return failure();
    // }
    // if (lhsType.getElementType() != rhsType.getElementType()) {
    //   return failure();
    // }
    // if (op->getResult(0).getType().dyn_cast<RankedTensorType>().getRank() !=
    //     1) {
    //   return failure();
    // }
    // get op as dotGeneral
    // cast op to mhlo::dotGeneralOp
    mhlo::DotGeneralOp dotGeneralOp =
        static_cast<mhlo::DotGeneralOp>(op.getOperation());
    if (!dotGeneralOp) {
      return failure();
    }

    auto dotDimensionNumbers = dotGeneralOp.getDotDimensionNumbers();
    // assert(dotDimensionNumbers.getLhsContractingDimensions().size() == 1);
    // assert(dotDimensionNumbers.getRhsContractingDimensions().size() == 1);
    int64_t lhsContractingDimension =
        dotDimensionNumbers.getLhsContractingDimensions()[0];
    int64_t rhsContractingDimension =
        dotDimensionNumbers.getRhsContractingDimensions()[0];
    auto lhsContractingDimensions =
        dotDimensionNumbers.getLhsContractingDimensions();
    auto rhsContractingDimensions =
        dotDimensionNumbers.getRhsContractingDimensions();
    auto lhsBatchingDimensions = dotDimensionNumbers.getLhsBatchingDimensions();
    auto rhsBatchingDimensions = dotDimensionNumbers.getRhsBatchingDimensions();

    //   if(lhsType.getRank() == 1  && rhsType.getRank() != 2){
    //             return failure();
    //         }

    // check if lhs contracting or rhs contracting is not 1
    if (lhsContractingDimension != 1 && rhsContractingDimension != 1) {
      // if rank is not 2
      //  if(lhsType.getRank() == 1  && rhsType.getRank() != 2){
      //      return failure();
      //  }
      //   if(lhsType.getRank() == 2  && rhsType.getRank() != 1){
      //      return failure();
      //  }

      return failure();
    }

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Type resultType = op.getResult().getType();

    // TensorType seedOrOffsetType =
    //     RankedTensorType::get({}, rewriter.getI64Type());

    // ModuleOp module = op->getParentRegion()->getParentOfType<ModuleOp>();
    // auto functionType = FunctionType::get(module.getContext(), {},
    //                                       ArrayRef<Type>{seedOrOffsetType});
    // func::FuncOp getSeedFunc = getOrCreatePrivateFunctionDeclare(
    //     module, "GetSeedFunc", "GetSeed", functionType);
    // func::FuncOp nextOffsetFunc = getOrCreatePrivateFunctionDeclare(
    //     module, "NextOffsetFunc", "NextOffset", functionType);

    // avoid to call @getSeed every time
    // auto getSeedOp = getOrCreateCallGetSeedOp(
    //     op->getParentRegion()->getParentOfType<func::FuncOp>(), getSeedFunc,
    //     rewriter);
    // auto getOffsetOp = rewriter.create<func::CallOp>(
    //     op->getLoc(), nextOffsetFunc, ArrayRef<Value>{});
    SmallVector<Value> bufferArgs{lhs, rhs};
    // if (!op.getType().hasStaticShape()) {
    //   bufferArgs.emplace_back(shape);
    // }
    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("device"),
                              rewriter.getStringAttr("cpu"));
    // auto byteir_attrs =
    //     op->template getAttrOfType<DictionaryAttr>(getCustomCallAttrName());
    auto attrs = getDefaultAttrs(rewriter);
    // attrs.emplace_back(rewriter.getStringAttr("lhs_contracting_dimension"),
    //                    rewriter.getI64IntegerAttr(lhsContractingDimension));
    // attrs.emplace_back(rewriter.getStringAttr("rhs_contracting_dimension"),
    //                       rewriter.getI64IntegerAttr(rhsContractingDimension));
    // attrs.emplace_back(rewriter.getStringAttr("lhs_batching_dimensions"),
    //                         rewriter.getI64ArrayAttr(lhsBatchingDimensions));
    // attrs.emplace_back(rewriter.getStringAttr("rhs_batching_dimensions"),
    //                         rewriter.getI64ArrayAttr(rhsBatchingDimensions));

    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getGemvhbmpimName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto computeOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    computeOp->setAttr("device", rewriter.getStringAttr("cpu"));

    //   computeOp->setAttr("lhs_contracting_dimension",
    //                      rewriter.getI64IntegerAttr(lhsContractingDimension));
    //   computeOp->setAttr("rhs_contracting_dimension",
    //                      rewriter.getI64IntegerAttr(rhsContractingDimension));
    //   computeOp->setAttr("lhs_batching_dimensions",
    //                      rewriter.getI64ArrayAttr(lhsBatchingDimensions));
    //   computeOp->setAttr("rhs_batching_dimensions",
    //                      rewriter.getI64ArrayAttr(rhsBatchingDimensions));
    computeOp->setAttr("device", rewriter.getStringAttr("cpu"));
    auto dimensionNumbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhsBatchingDimensions=*/{lhsBatchingDimensions},
        /*rhsBatchingDimensions=*/{rhsBatchingDimensions},
        {lhsContractingDimension}, {rhsContractingDimension});

    computeOp->setAttr("dimension_numbers", dimensionNumbers);

    rewriter.replaceOp(op, computeOp->getResults());
    return success();
  }
};

// dot to gemv

// struct ConvertDotToGEMVCustomOp : public OpRewritePattern<mhlo::DotOp> {
// public:
//   using OpRewritePattern<mhlo::DotOp>::OpRewritePattern;
//   LogicalResult matchAndRewrite(mhlo::DotOp op,
//                                 PatternRewriter &rewriter) const {
//     // check if dot can be converted to gemv
//     if (op->getNumOperands() != 2) {
//       return failure();
//     }
//     auto lhsType = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
//     auto rhsType = op->getOperand(1).getType().dyn_cast<RankedTensorType>();
//     if (!lhsType || !rhsType) {
//       return failure();
//     }
//     if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
//       return failure();
//     }
//     if (lhsType.getShape()[1] != rhsType.getShape()[0]) {
//       return failure();
//     }
//     if (lhsType.getElementType() != rhsType.getElementType()) {
//       return failure();
//     }
//     if (op->getResult(0).getType().dyn_cast<RankedTensorType>().getRank() !=
//         0) {
//       return failure();
//     }

//     Value lhs = op.getOperand(0);
//     Value rhs = op.getOperand(1);
//     // Value lhsPromoted = promoteType(op->getLoc(), lhs, resultType,
//     rewriter);
//     // Value rhsPromoted = promoteType(op->getLoc(), rhs, resultType,
//     rewriter);
//     // auto shape = op.getShape();
//     Type resultType = op.getResult().getType();
//     TensorType seedOrOffsetType =
//         RankedTensorType::get({}, rewriter.getI64Type());

//     ModuleOp module = op->getParentRegion()->getParentOfType<ModuleOp>();
//     auto functionType = FunctionType::get(module.getContext(), {},
//                                           ArrayRef<Type>{seedOrOffsetType});
//     func::FuncOp getSeedFunc = getOrCreatePrivateFunctionDeclare(
//         module, "GetSeedFunc", "GetSeed", functionType);
//     func::FuncOp nextOffsetFunc = getOrCreatePrivateFunctionDeclare(
//         module, "NextOffsetFunc", "NextOffset", functionType);

//     // avoid to call @getSeed every time
//     auto getSeedOp = getOrCreateCallGetSeedOp(
//         op->getParentRegion()->getParentOfType<func::FuncOp>(), getSeedFunc,
//         rewriter);
//     auto getOffsetOp = rewriter.create<func::CallOp>(
//         op->getLoc(), nextOffsetFunc, ArrayRef<Value>{});
//     SmallVector<Value> bufferArgs{lhs, rhs, getSeedOp.getResults()[0],
//                                   getOffsetOp.getResults()[0]};
//     // if (!op.getType().hasStaticShape()) {
//     //   bufferArgs.emplace_back(shape);
//     // }
//     auto dictAttr =
//         op->template getAttrOfType<DictionaryAttr>(getCustomCallAttrName());

//     auto attrs = getDefaultAttrs(rewriter);
//     attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
//                        rewriter.getStringAttr(getGemvUpmemName()));
//     attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
//                        dictAttr);
//     auto newOp = rewriter.create<mhlo::CustomCallOp>(
//         op->getLoc(), resultType, bufferArgs,
//         ArrayRef<NamedAttribute>(attrs));
//     rewriter.replaceOp(op, newOp->getResults());
//   }
// };
template <typename DerivedT>
class ConvertOpToPimCustomCallBase : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = ConvertOpToPimCustomCallBase;

  ConvertOpToPimCustomCallBase()
      : ::mlir::OperationPass<ModuleOp>(::mlir::TypeID::get<DerivedT>()) {}
  ConvertOpToPimCustomCallBase(const ConvertOpToPimCustomCallBase &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("convert-op-to-pimcustomcall");
  }
  ::llvm::StringRef getArgument() const override {
    return "convert-op-to-pimcustomcall";
  }

  ::llvm::StringRef getDescription() const override {
    return "Convert op to mhlo.custom_call";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ConvertOpToPimCustomCall");
  }
  ::llvm::StringRef getName() const override {
    return "ConvertOpToPimCustomCall";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertOpToPimCustomCallBase<DerivedT>)

protected:
  ::mlir::Pass::Option<std::string> anchorTag{
      *this, "anchor-tag",
      ::llvm::cl::desc("Optional unitAttr anchored tag to apply this pass")};
};

struct ConvertOpToPimCustomCallPass
    : public ConvertOpToPimCustomCallBase<ConvertOpToPimCustomCallPass> {

  ConvertOpToPimCustomCallPass(llvm::StringRef anchor)
      : ConvertOpToPimCustomCallBase() {
    this->anchorTag = anchor.str();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!this->anchorTag.empty() && !funcOp->hasAttr(this->anchorTag)) {
        continue;
      }

      MLIRContext *context = &getContext();

      RewritePatternSet patterns(context);

      populateGemvRewritePattern(patterns);

      FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

void mlir::populateGemvRewritePattern(RewritePatternSet &patterns) {
  //   patterns.add<ConvertDotToGEMVCustomOp>(patterns.getContext());
  patterns.add<ConvertDotGeneralToGEMVCustomOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertOpToPimCustomCallPass(llvm::StringRef anchor) {
  return std::make_unique<ConvertOpToPimCustomCallPass>(anchor);
}