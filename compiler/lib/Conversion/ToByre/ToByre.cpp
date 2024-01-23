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
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/HashUtils.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
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
        rewriter.replaceOpWithNewOp<byre::ComputeOp>(
            op, TypeRange{}, key, operands, memoryEffectsAttr);

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
            if (src.getType() != dst.getType()) {
              src = rewriter.create<byre::AliasOp>(op->getLoc(), dst.getType(),
                                                   src, 0);
            }
            rewriter.create<memref::CopyOp>(loc, src, dst);
          }
        }
      }
    }

    addAttrs(computeOp.getOperation(), attrs);

    return success();
  }
};
} // namespace

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
static void
identifyEntryPointFuncAndCalls(ModuleOp m,
                               llvm::SmallVector<func::FuncOp, 4> &entries,
                               llvm::SmallVector<func::CallOp, 16> &calls,
                               llvm::SetVector<func::FuncOp> &removeFuncs) {
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
        removeFuncs.insert(calleeFuncOp);
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
        constantValue;
    mlir::OpBuilder opBuilder(retOp);
    for (auto retValIter : llvm::enumerate(retOp.getOperands())) {
      auto retVal = retValIter.value();
      if (retVal.getDefiningOp<memref::GetGlobalOp>()) {
        // if return constant value, insert a memref.copy
        opBuilder.setInsertionPoint(retOp);
        opBuilder.create<memref::CopyOp>(
            retOp.getLoc(), retVal, func.getArgument(idx + retValIter.index()));
        if (constantValue.find(retVal) == constantValue.end()) {
          constantValue[retVal] = idx + retValIter.index();
        } else {
          // append byre.arg_alias_index to func op
          func.setArgAttr(idx + retValIter.index(),
                          ByreDialect::getEntryPointFuncArgAliasIndexAttrName(),
                          opBuilder.getI64IntegerAttr(constantValue[retVal]));
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
    if (!op->hasAttr(getByreCallOpReadonlyOperandNumAttrName()))
      op->erase();
  }
}

// TODO: it's really for lmhlo now?
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

static void replaceGetGlobalConstantWithFuncArgument(func::FuncOp funcOp) {
  SmallVector<std::pair<Type, DenseElementsAttr>> typeAndValue;
  SmallVector<Operation *> globalConstants;
  funcOp.walk([&](memref::GetGlobalOp getGlobalOp) {
    auto globalOp = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        getGlobalOp, getGlobalOp.getNameAttr());
    if (!globalOp)
      return;

    auto valueOrNot = globalOp.getInitialValue();
    if (!valueOrNot || !globalOp.getConstant()) {
      return;
    }

    DenseElementsAttr value =
        llvm::dyn_cast_or_null<DenseElementsAttr>(*valueOrNot);
    if (!value || value.isSplat()) {
      return;
    }

    typeAndValue.emplace_back(getGlobalOp.getResult().getType(), value);
    globalConstants.emplace_back(getGlobalOp);
  });

  auto argTypeAttrName = byre::ByreDialect::getEntryPointFuncArgTypeAttrName();
  auto argNameAttrName = byre::ByreDialect::getEntryPointFuncArgNameAttrName();
  auto argWeightValueAttrName =
      byre::ByreDialect::getEntryPointFuncArgWeightValueAttrName();
  auto context = funcOp->getContext();
  unsigned int weightCount = 0;
  auto oldFuncType = funcOp.getFunctionType();

  mlir::OpBuilder opBuilder(funcOp);
  llvm::SmallVector<DictionaryAttr, 4> newArgAttrs;
  llvm::SmallVector<Type, 16> newInputTypes;
  llvm::SmallVector<Type, 16> newOutputTypes(oldFuncType.getResults().begin(),
                                             oldFuncType.getResults().end());

  for (auto &item : typeAndValue) {
    newInputTypes.emplace_back(item.first);
    NamedAttrList argAttr;
    argAttr.append(
        argNameAttrName,
        StringAttr::get(context, Twine("Weight") + Twine(weightCount++)));
    argAttr.append(argTypeAttrName,
                   byre::EntryFuncArgTypeAttr::get(
                       context, byre::EntryFuncArgType::Weight));
    argAttr.append(argWeightValueAttrName, item.second);
    newArgAttrs.emplace_back(argAttr.getDictionary(funcOp->getContext()));
  }

  if (newInputTypes.size() == 0) {
    return;
  }

  if (funcOp.getArgAttrsAttr()) {
    auto oldArgAttrs = funcOp.getArgAttrsAttr().getAsRange<DictionaryAttr>();
    for (auto argAttr : oldArgAttrs) {
      newArgAttrs.emplace_back(argAttr);
    }
  }

  newInputTypes.insert(newInputTypes.end(), oldFuncType.getInputs().begin(),
                       oldFuncType.getInputs().end());
  mlir::FunctionType newFuncType =
      opBuilder.getFunctionType(newInputTypes, newOutputTypes);

  auto funcInterface = cast<FunctionOpInterface>(funcOp.getOperation());
  Block &entry = funcInterface->getRegion(0).front();
  funcInterface.setFunctionTypeAttr(TypeAttr::get(newFuncType));

  for (int i = 0; i < static_cast<int>(typeAndValue.size()); ++i) {
    entry.insertArgument(i, newInputTypes[i], funcOp->getLoc());
  }

  mlir::function_interface_impl::setAllArgAttrDicts(funcOp, newArgAttrs);
  int idx = 0;
  for (auto op : globalConstants) {
    auto value = op->getResult(0);
    value.replaceAllUsesWith(entry.getArgument(idx));
    idx += 1;
    op->erase();
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

  if (mlir::failed(runPipeline(pm, m))) {
    signalPassFailure();
  }
}

void ConvertFuncAndCallToByrePass::runOnOperation() {
  ModuleOp m = getOperation();
  MLIRContext &ctx = getContext();
  llvm::SmallVector<func::FuncOp, 4> entryCollector;
  llvm::SmallVector<func::CallOp, 16> callCollector;
  llvm::SetVector<func::FuncOp> removeFuncCollector;

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

    replaceGetGlobalConstantWithFuncArgument(func);

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

  for (auto func : removeFuncCollector.takeVector()) {
    func->erase();
  }
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
