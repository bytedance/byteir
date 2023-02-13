//===- ConvertFuncToCustomCall.cpp ----------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/ConvertFuncToCustomCall.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace mlir::mhlo;

namespace {

// TODO: move to util
static std::string getFuncName(const std::string &str) {
  size_t firtDot = str.find(".");
  if (firtDot != std::string::npos) {
    size_t secondDot = str.find(".", firtDot + 1);
    return str.substr(0, secondDot);
  }
  return "";
}

class ConvertFuncToCustomCallPass : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = ConvertFuncToCustomCallPass;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFuncToCustomCallPass)

  ConvertFuncToCustomCallPass()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<ConvertFuncToCustomCallPass>()) {}

  ConvertFuncToCustomCallPass(const ConvertFuncToCustomCallPass &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  explicit ConvertFuncToCustomCallPass(
      FuncToCustomCallConverterBase *externalConverter)
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<ConvertFuncToCustomCallPass>()),
        converter(externalConverter) {}

  // Note command-line was disable in this pass, due to it using a class to
  // drive Please use TestConvertFuncToCustomCall (test-convert-func-to-custom)
  // in command-line
#if 0 
  /// Returns the command-line argument attached to this pass.
   static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("convert-func-to-custom");
  }
  ::llvm::StringRef getArgument() const override { return "convert-func-to-custom"; }

  ::llvm::StringRef getDescription() const override {
    return "Convert Func to CustomCall";
  }
#endif

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ConvertFuncToCustomCall");
  }
  ::llvm::StringRef getName() const override {
    return "ConvertFuncToCustomCall";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() ==
           ::mlir::TypeID::get<ConvertFuncToCustomCallPass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertFuncToCustomCallPass>(
        *static_cast<const ConvertFuncToCustomCallPass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override;

protected:
  FuncToCustomCallConverterBase *converter = nullptr;
};

} // namespace

std::function<void(func::FuncOp, ModuleOp)>
mlir::FuncToCustomCallConverterLookup::getCustomizedConversion(func::FuncOp f) {
  const std::string &orig = f.getName().str();
  auto it = funcNameToCustomizedConversion.find(getFuncName(orig));
  return it == funcNameToCustomizedConversion.end() ? nullptr : it->second;
}

bool mlir::FuncToCustomCallConverterLookup::checkFunc(func::FuncOp f) {
  const std::string &orig = f.getName().str();
  return funcNameToCustomMeta.count(getFuncName(orig)) > 0 ||
         funcNameToCustomizedConversion.count(getFuncName(orig)) > 0;
}

NamedAttrList FuncToCustomCallConverterLookup::getAttrs(func::FuncOp func) {
  auto ctx = func.getContext();
  const std::string &orig = func.getName().str();
  const auto &meta = funcNameToCustomMeta[getFuncName(orig)];

  NamedAttrList attrs;
  attrs.append(::llvm::StringRef("call_target_name"),
               StringAttr::get(ctx, meta.callTargetName));

  attrs.append(::llvm::StringRef("has_side_effect"),
               BoolAttr::get(ctx, meta.hasSideEffect));

  // the rest ones use default values
  attrs.append(::llvm::StringRef("backend_config"), StringAttr::get(ctx));

  attrs.append(::llvm::StringRef("api_version"),
               CustomCallApiVersionAttr::get(
                   ctx, CustomCallApiVersion::API_VERSION_ORIGINAL));

  attrs.append(::llvm::StringRef("called_computations"),
               ArrayAttr::get(ctx, {}));

  return attrs;
}

TypeRange
mlir::FuncToCustomCallConverterLookup::getResultTypes(func::FuncOp func) {
  const std::string &orig = func.getName().str();
  const auto &meta = funcNameToCustomMeta[getFuncName(orig)];

  if (meta.useDefault) {
    return func.getResultTypes();
  }

  SmallVector<Type> newResults;
  for (auto id : meta.resultOldIndices) {
    newResults.push_back(func.getResultTypes()[id]);
  }
  return newResults;
}

ValueRange
mlir::FuncToCustomCallConverterLookup::getOperands(func::CallOp call) {
  const std::string &orig = call.getCallee().str();
  const auto &meta = funcNameToCustomMeta[getFuncName(orig)];

  if (meta.useDefault) {
    return call.getOperands();
  }

  SmallVector<Value> newOperands;
  for (auto id : meta.opernadOldIndices) {
    newOperands.push_back(call.getOperand(id));
  }
  return newOperands;
}

unsigned
mlir::FuncToCustomCallConverterLookup::getNewResultIdx(func::CallOp call,
                                                       unsigned oldIdx) {
  const std::string &orig = call.getCallee().str();
  const auto &meta = funcNameToCustomMeta[getFuncName(orig)];

  if (meta.useDefault) {
    return oldIdx;
  }

  return meta.resultNewIndices[oldIdx];
}

void ConvertFuncToCustomCallPass::runOnOperation() {

  auto m = getOperation();
  auto ctx = m.getContext();

  // early return if no converter
  if (nullptr == converter) {
    return;
  }

  SmallVector<func::FuncOp> funcCollecter;
  SmallVector<Operation *> opEraser;

  // iterte all func
  for (auto func : m.getOps<func::FuncOp>()) {
    if (converter->checkFunc(func)) {
      funcCollecter.push_back(func);
    }
  }

  for (auto func : funcCollecter) {
    auto convertLogic = converter->getCustomizedConversion(func);
    if (convertLogic != nullptr) {
      convertLogic(func, m);
      continue;
    }

    auto resultTys = converter->getResultTypes(func);
    auto attrs = converter->getAttrs(func);

    // insert convert for a call
    auto maybeSymbolUses = func.getSymbolUses(m);
    OpBuilder b(ctx);
    for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
      if (auto callOp = dyn_cast<CallOp>(symbolUse.getUser())) {
        auto loc = callOp.getLoc();
        b.setInsertionPoint(callOp);
        auto operands = converter->getOperands(callOp);

        opEraser.push_back(callOp);

        auto custom =
            b.create<mhlo::CustomCallOp>(loc, resultTys, operands, attrs);

        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          auto newIdx = converter->getNewResultIdx(callOp, i);
          callOp.getResult(i).replaceAllUsesWith(custom.getResult(newIdx));
        }
      } // endif callOp
    }   // endfor symbolUse
  }     // endfor func

  // remove replaced calls first
  for (auto *op : opEraser) {
    op->erase();
  }

  // then remove replaced func
  for (auto func : funcCollecter) {
    func.erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToCustomCallPass(
    FuncToCustomCallConverterBase *converter) {
  return std::make_unique<ConvertFuncToCustomCallPass>(converter);
}
