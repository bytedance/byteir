//===- DTypeConversion.cpp -----------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/DTypeConversion.h"
#include "byteir/Utils/AttrUtils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace mlir::mhlo;

namespace {

class DTypeConversionPass : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = DTypeConversionPass;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DTypeConversionPass)

  DTypeConversionPass()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<DTypeConversionPass>()) {}

  DTypeConversionPass(const DTypeConversionPass &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  explicit DTypeConversionPass(DTypeConvertRuleBase *externalCollector)
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<DTypeConversionPass>()),
        collector(externalCollector) {}

  ::llvm::StringRef getDescription() const override {
    return "Convert data types for FuncOp and Ops inside FuncOp";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("DTypeConversionPass");
  }
  ::llvm::StringRef getName() const override { return "DTypeConversionPass"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DTypeConversionPass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DTypeConversionPass>(
        *static_cast<const DTypeConversionPass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override;

protected:
  DTypeConvertRuleBase *collector = nullptr;
};

struct DTypeOpConversionPattern : public RewritePattern {
  DTypeOpConversionPattern(
      MLIRContext *context,
      llvm::DenseMap<
          llvm::StringRef,
          std::vector<std::pair<std::vector<Type>, std::vector<Type>>>>
          &convertRule)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context), rule(convertRule) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef opName = op->getName().getStringRef();
    MLIRContext *context = op->getContext();
    if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
      opName = customCall.getCallTargetName();
    }
    // don't insert convert op if it's not in the rule
    if (rule.count(opName) == 0)
      return failure();

    // Use most preferred types for now
    assert(rule[opName].size() > 0);
    auto ioTy = rule[opName][0];
    assert(ioTy.first.size() == op->getNumOperands());
    assert(ioTy.second.size() == op->getNumResults());
    bool changed = false;
    bool hasRegion = op->getNumRegions() > 0;
    if (hasRegion) {
      // TODO: handle op that has multiple regions
      Block &body = op->getRegion(0).front();
      assert(body.getArguments().size() == op->getNumOperands());
      auto returnOp = body.getTerminator();
      assert(returnOp->getNumOperands() == op->getNumResults());
    }
    rewriter.setInsertionPoint(op);
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      auto originalTy = getElementTypeOrSelf(op->getOperand(i).getType());
      if (originalTy != ioTy.first[i]) {
        changed = true;
        auto oldType = op->getOperand(i).getType();
        assert(ioTy.first[i]);
        auto newTy =
            cast<TensorType>(oldType).cloneWith(std::nullopt, ioTy.first[i]);
        auto convert = rewriter.create<mhlo::ConvertOp>(op->getLoc(), newTy,
                                                        op->getOperand(i));
        convert->setAttr(getConvertAnchorName(), UnitAttr::get(context));
        op->getOpOperand(i).set(convert.getResult());
        if (hasRegion) {
          Block &body = op->getRegion(0).front();
          auto arg = body.getArgument(i);
          auto newArgTy = cast<TensorType>(arg.getType())
                              .cloneWith(std::nullopt, ioTy.first[i]);
          arg.setType(newArgTy);
        }
      }
    }
    rewriter.setInsertionPointAfter(op);
    for (size_t i = 0; i < op->getNumResults(); ++i) {
      auto originalTy = getElementTypeOrSelf(op->getResult(i).getType());
      if (originalTy != ioTy.second[i]) {
        changed = true;
        auto oldTy = op->getResult(i).getType();
        auto newTy = cast<TensorType>(op->getResult(i).getType())
                         .cloneWith(std::nullopt, ioTy.second[i]);
        op->getResult(i).setType(newTy);
        auto convert = rewriter.create<mhlo::ConvertOp>(op->getLoc(), oldTy,
                                                        op->getResult(i));
        convert->setAttr(getConvertAnchorName(), UnitAttr::get(context));
        op->getResult(i).replaceAllUsesExcept(convert.getResult(), convert);
        if (hasRegion) {
          Block &body = op->getRegion(0).front();
          auto returnOp = body.getTerminator();
          auto res = returnOp->getOperand(i);
          res.setType(cast<TensorType>(res.getType())
                          .cloneWith(std::nullopt, ioTy.second[i]));
        }
      }
    }
    return changed ? success() : failure();
  }

private:
  llvm::DenseMap<llvm::StringRef,
                 std::vector<std::pair<std::vector<Type>, std::vector<Type>>>>
      &rule;
};

LogicalResult foldConsecutiveConvertOp(mhlo::ConvertOp op,
                                       PatternRewriter &rewriter) {
  if (!llvm::isa_and_nonnull<mhlo::ConvertOp>(
          op.getOperand().getDefiningOp())) {
    return failure();
  }
  if (!op->hasAttrOfType<UnitAttr>(getConvertAnchorName()))
    return failure();

  mhlo::ConvertOp firstConvert =
      cast<mhlo::ConvertOp>(op.getOperand().getDefiningOp());
  if (!firstConvert->hasAttrOfType<UnitAttr>(getConvertAnchorName()))
    return failure();
  auto input = firstConvert.getOperand();
  auto inputTy = getElementTypeOrSelf(input);
  // fp64->fp32->fp16 to fp64->fp16 is not handled here because it's handled
  // already by the upstream Only fold the case where two convert can be
  // cancelled
  if (inputTy == getElementTypeOrSelf(op.getResult())) {
    // cancel second convert
    rewriter.replaceOp(op, input);
    return success();
  }
  return failure();
}

} // namespace

void DTypeConversionPass::runOnOperation() {
  auto m = getOperation();

  // early return if no collector
  if (nullptr == collector) {
    return;
  }
  // iterate all func
  for (auto func : m.getOps<func::FuncOp>()) {
    if (!collector->checkFunc(func)) {
      continue;
    }

    RewritePatternSet patterns(func->getContext());
    patterns.add<DTypeOpConversionPattern>(func->getContext(),
                                           collector->convertRules);

    patterns.add(foldConsecutiveConvertOp);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

    func.walk([&](mhlo::ConvertOp op) {
      // remove convert anchor attr
      if (op->hasAttrOfType<UnitAttr>(getConvertAnchorName())) {
        op->removeAttr(getConvertAnchorName());
      }
    });
    if (collector->canModifyFuncArg(func)) {
      auto &body = func.getBody().front();
      bool changed = false;

      FunctionType funcType = func.getFunctionType();
      mlir::SmallVector<Type> argTypes;
      argTypes.reserve(funcType.getNumInputs());
      mlir::SmallVector<Type> retTypes;
      retTypes.reserve(funcType.getNumResults());

      for (auto arg : body.getArguments()) {
        if (arg.hasOneUse() && isa<mhlo::ConvertOp>(*arg.getUsers().begin())) {
          auto convert = cast<mhlo::ConvertOp>(*arg.getUsers().begin());
          auto newTy = convert.getResult().getType();
          convert.getResult().replaceAllUsesWith(arg);
          arg.setType(newTy);
          changed = true;
        }
        argTypes.push_back(arg.getType());
      }

      auto returnOp =
          mlir::dyn_cast<mlir::func::ReturnOp>(body.getTerminator());
      for (auto operand : returnOp->getOperands()) {
        if (auto convert = dyn_cast<mhlo::ConvertOp>(operand.getDefiningOp())) {
          auto newTy = convert.getOperand().getType();
          returnOp->replaceUsesOfWith(operand, convert.getOperand());
          retTypes.push_back(newTy);
          changed = true;
        } else {
          retTypes.push_back(operand.getType());
        }
      }
      if (changed) {
        // rewrite FunctionType
        func.setType(FunctionType::get(func.getContext(), argTypes, retTypes));
      }
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createDTypeConversionPass(DTypeConvertRuleBase *collector) {
  return std::make_unique<DTypeConversionPass>(collector);
}
