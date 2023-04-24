//===- FuncArgRearrangement.cpp -------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/FuncArgRearrangement.h"

#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/FuncUtils.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include "./PassDetail.h"

#define DEBUG_TYPE "func-arg-rearrange"

using namespace mlir;
using namespace llvm;

namespace {

static bool checkUser(Value val, Operation *user,
                      SmallPtrSetImpl<Operation *> &indicators) {

  if (user == nullptr)
    return false;

  for (auto result : user->getResults()) {
    if (val == result) {
      return true;
    }
  }

  for (auto result : user->getResults()) {
    for (auto grandUser : result.getUsers()) {
      // avoid going through another indicator
      if (indicators.contains(grandUser)) {
        continue;
      }

      if (checkUser(val, grandUser, indicators)) {
        return true;
      }
    }
  }

  return false;
}

static std::optional<unsigned>
findMostLikelyUse(Value val, Operation *user,
                  ArrayRef<Operation *> indicators) {

  for (auto indicator : indicators) {
    if (user == indicator) {
      // check direct use of indicator
      for (unsigned i = 0; i < indicator->getNumOperands(); ++i) {
        if (indicator->getOperand(i) == val) {
          LLVM_DEBUG(llvm::dbgs()
                     << "findMostLikelyUse found direct user " << i << "\n");
          return i;
        }
      }
    }
  }

  SmallPtrSet<Operation *, 4> allIndicatorSet(indicators.begin(),
                                              indicators.end());
  for (auto indicator : indicators) {
    // check indirect use through user
    for (unsigned i = 0; i < indicator->getNumOperands(); ++i) {
      if (checkUser(indicator->getOperand(i), user, allIndicatorSet)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "findMostLikelyUse found indirect user " << i << "\n");
        return i;
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "findMostLikelyUse found no idx\n");
  return std::nullopt;
}

class FuncArgRearrangementPass : public ::mlir::OperationPass<ModuleOp> {
public:
  using Base = FuncArgRearrangementPass;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuncArgRearrangementPass)

  FuncArgRearrangementPass()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<FuncArgRearrangementPass>()) {}

  FuncArgRearrangementPass(const FuncArgRearrangementPass &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  FuncArgRearrangementPass(FuncArgRearrangerBuilderBase *builder,
                           const std::string &anchor, bool keepAnchor)
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<FuncArgRearrangementPass>()),
        rearrangeBuilder(builder), anchorAttr(anchor), keepAnchor(keepAnchor) {}

// Note command-line was disable in this pass, due to it using a class to drive
// Please use TestConvertInsertion (test-insert-convert) in command-line
#if 0 
  /// Returns the command-line argument attached to this pass.
   static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("rearrange-func-arg");
  }
  ::llvm::StringRef getArgument() const override { return "rearrange-func-arg"; }

  ::llvm::StringRef getDescription() const override {
    return "Func Arg Rearrangement: pack, reorder args and returns of func";
  }
#endif

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("FuncArgRearrangement");
  }
  ::llvm::StringRef getName() const override { return "FuncArgRearrangement"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<FuncArgRearrangementPass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<FuncArgRearrangementPass>(
        *static_cast<const FuncArgRearrangementPass *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override;

protected:
  FuncArgRearrangerBuilderBase *rearrangeBuilder = nullptr;
  std::string anchorAttr = "";
  bool keepAnchor;
};

void FuncArgRearrangementPass::runOnOperation() {
  if (rearrangeBuilder == nullptr || anchorAttr.empty())
    return;

  ModuleOp m = getOperation();

  // collect all func
  SmallVector<func::FuncOp> targetFuncs;
  for (auto f : m.getOps<func::FuncOp>()) {
    if (f->hasAttr(anchorAttr)) {
      targetFuncs.push_back(f);
    }
  }

  llvm::DenseMap<Value, SmallVector<Value>> duplicateReverseMap;
  llvm::SmallVector<func::FuncOp> newFuncs;

  for (auto f : targetFuncs) {
    SmallVector<Operation *> eraser;

    auto rearrangerPtr = rearrangeBuilder->createFuncArgRearranger(f);
    // skip if nullptr or init failed
    if (rearrangerPtr == nullptr || !rearrangerPtr->init()) {
      continue;
    }

    // 1. Create a new Func
    OpBuilder builder(f);
    auto newFunc = rearrangerPtr->getOrCreateNewFunc(builder);
    newFuncs.push_back(newFunc);

    if (keepAnchor)
      cloneAllExtraFuncAttrs(f, newFunc);
    else
      cloneAllExtraFuncAttrs(f, newFunc, {anchorAttr});

    // 2. Rewrite Body if Func is non-empty
    if (!f.empty()) {
      // handle args
      auto entry = newFunc.addEntryBlock();
      builder.setInsertionPointToEnd(entry);

      // assign IRMapping from oldArg to newVal
      IRMapping argBvm;

      auto newArgs = llvm::to_vector(llvm::map_range(
          newFunc.getArguments(),
          [&](const BlockArgument &val) -> Value { return val; }));

      for (unsigned i = 0; i < f.getNumArguments(); ++i) {
        auto toValList =
            rearrangerPtr->getOrCreateOldFromNewFuncArg(builder, i, newArgs);
        if (!toValList.empty()) {
          if (toValList.size() > 1) {
            duplicateReverseMap.try_emplace(toValList.front(), toValList);
          }

          argBvm.map(f.getArgument(i), toValList.front());
        }
      }

      // clone body by replace oldArgs to newVals
      f.getBody().cloneInto(&newFunc.getBody(), argBvm);

      // update duplicateReverseMap if needed
      for (auto &it : duplicateReverseMap) {
        if (argBvm.contains(it.first)) {
          auto toVal = argBvm.lookup(it.first);
          it.first = toVal;
        }

        for (auto &v : it.second) {
          if (argBvm.contains(v)) {
            auto toVal = argBvm.lookup(v);
            v = toVal;
          }
        }
      }

      collapseFuncRegion(newFunc);

      // handle ReturnOp
      auto oldRet =
          cast<func::ReturnOp>(newFunc.getBody().back().getTerminator());
      builder.setInsertionPoint(oldRet);
      auto oldRetOperands = llvm::to_vector(oldRet.getOperands());
      SmallVector<Value> newRetOperands;
      for (unsigned i = 0; i < newFunc.getNumResults(); ++i) {
        auto newVal = rearrangerPtr->getOrCreateNewFromOldFuncResult(
            builder, i, oldRetOperands);
        newRetOperands.push_back(newVal);
      }

      builder.create<func::ReturnOp>(oldRet.getLoc(), newRetOperands);

      // collect old ReturnOp in eraser
      eraser.push_back(oldRet);
    } // f.empty

    // 3. Rewrite Call
    auto maybeSymbolUses = f.getSymbolUses(m);
    for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
      if (auto callOp = dyn_cast<func::CallOp>(symbolUse.getUser())) {

        builder.setInsertionPoint(callOp);

        // handle callOp's args
        auto oldArgs = llvm::to_vector(callOp.getOperands());
        SmallVector<Value> newArgs;
        for (unsigned i = 0; i < newFunc.getNumArguments(); ++i) {
          auto newArg =
              rearrangerPtr->getOrCreateNewFromOldFuncArg(builder, i, oldArgs);
          newArgs.push_back(newArg);
        }

        auto newCall =
            builder.create<func::CallOp>(callOp.getLoc(), newFunc, newArgs);

        // handle callOp's results
        auto newCallResults = llvm::to_vector(
            llvm::map_range(newCall.getResults(),
                            [&](const OpResult &val) -> Value { return val; }));

        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
          auto toResultList = rearrangerPtr->getOrCreateOldFromNewFuncResult(
              builder, i, newCallResults);
          if (!toResultList.empty()) {
            if (toResultList.size() > 1) {
              duplicateReverseMap.try_emplace(toResultList.front(),
                                              toResultList);
            }

            callOp.getResult(i).replaceAllUsesWith(toResultList.front());
          }
        }

        // collect old callOp in eraser
        eraser.push_back(callOp);
      }
    } // endfor maybeSymbolUses

    eraser.push_back(f);

    // erase all ops in eraser
    for (auto op : eraser) {
      op->erase();
    }
  } // endfor targetFuncs

  // the following handle duplicated mapping using a heuristic ordering argument
  // TODO: (LWC) added more advanced method later from a config object

  // use call and return as indicator to resolve duplication
  llvm::SmallVector<Operation *> duplicateResolveIndicator;
  for (auto f : newFuncs) {
    // collect returns
    if (!f.empty()) {
      duplicateResolveIndicator.push_back(f.getBody().back().getTerminator());
    }

    // collect calls
    auto maybeSymbolUses = f.getSymbolUses(m);
    for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
      if (auto callOp = dyn_cast<func::CallOp>(symbolUse.getUser())) {
        duplicateResolveIndicator.push_back(callOp);
      }
    }
  }

  for (auto &it : duplicateReverseMap) {
    // find all replacable user
    SmallVector<std::pair<unsigned, Operation *>> useIdxAndOp;
    for (auto user : it.first.getUsers()) {
      auto maybeIdx =
          findMostLikelyUse(it.first, user, duplicateResolveIndicator);
      if (maybeIdx.has_value()) {
        useIdxAndOp.emplace_back(*maybeIdx, user);
      }
    }

    // sort by idx
    llvm::stable_sort(useIdxAndOp, [&](std::pair<unsigned, Operation *> lhs,
                                       std::pair<unsigned, Operation *> rhs) {
      return lhs.first < rhs.first;
    });

    unsigned cnt = 0;
    for (auto &p : useIdxAndOp) {
      if (cnt == 0) {
        cnt++;
        continue;
      }
      if (cnt >= it.second.size()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "more replacable value than duplicated value\n");
        break;
      }

      // replace specific op
      it.first.replaceUsesWithIf(it.second[cnt], [&](OpOperand &use) {
        return use.getOwner() == p.second;
      });
      cnt++;
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createFuncArgRearrangementPass(FuncArgRearrangerBuilderBase *builder,
                                     const std::string &anchor,
                                     bool keepAnchor) {
  return std::make_unique<FuncArgRearrangementPass>(builder, anchor,
                                                    keepAnchor);
}
