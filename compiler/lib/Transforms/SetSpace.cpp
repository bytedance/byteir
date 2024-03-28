//===- SetSpace.cpp -------------------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/SetSpace.h"

#include "byteir/Analysis/SideEffect.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#include "./PassDetail.h"

#define DEBUG_TYPE "set-space-passes"
#define SPACE_ATTR_NAME "device"

using namespace byteir;
using namespace mlir;
using namespace mlir::func;
using namespace mlir::memref;

namespace {

// local common types
using UpdateFuncType_t = std::pair<SmallVector<Type, 4>, SmallVector<Type, 4>>;
using CopyType_t = std::pair<Value, Attribute>;
using GlobalType_t = std::pair<Operation *, Attribute>;

// local utils
bool isMemref(Operation &op) {
  Dialect *dialect = op.getDialect();
  if (isa<memref::GetGlobalOp>(op))
    return false;

  return dialect && isa<MemRefDialect>(dialect);
}

const std::string &getSpace(ArrayRef<std::string> spaces, size_t offset) {
  if (offset < spaces.size()) {
    return spaces[offset];
  }
  return spaces.back();
}

bool isEmptyStringAttr(Attribute attr) {
  if (auto strAttr = attr.dyn_cast_or_null<StringAttr>()) {
    return strAttr.strref().empty();
  }
  return false;
}

bool isFuncCorrectSpace(func::FuncOp func, size_t offset, Attribute space,
                        bool isArg) {
  FunctionType funcType = func.getFunctionType();
  Type argType;
  if (isArg) {
    argType = funcType.getInput(offset);
  } else {
    argType = funcType.getResult(offset);
  }

  if (auto memRefTy = argType.dyn_cast<MemRefType>()) {
    return memRefTy.getMemorySpace() == space;
  }
  return false;
}

bool isFuncNotCompatiableWithSpace(func::FuncOp func, Attribute space) {
  if (auto funcSpaceAttr = func->getAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
    return funcSpaceAttr != space;
  }

  // func has not space attr
  // check whehther is public
  return func.isPublic();
}

// Maybe change this to bind Module Op later
Attribute getOrCreateSpaceAttr(ModuleOp m, llvm::StringRef name) {
  return StringAttr::get(m.getContext(), name);
}

memref::AllocOp createNewAllocWithDstMemrefTy(OpBuilder &b, Location loc,
                                              Value input,
                                              MemRefType dstMemrefTy) {
  // if the definingOp of input  is also a AllocOp,
  // reuse the operand to avoid redudant dimOp under dynamic shape.
  if (auto preAlloc = input.getDefiningOp<memref::AllocOp>()) {
    return b.create<memref::AllocOp>(loc, dstMemrefTy, preAlloc.getOperands());
  }

  int64_t rank = dstMemrefTy.getRank();
  SmallVector<Value> dynamicDims;
  // Get the dynamic dims of dstType
  for (int i = 0; i < rank; ++i) {
    if (!dstMemrefTy.isDynamicDim(i))
      continue;
    Value index = b.create<arith::ConstantIndexOp>(loc, i);
    Value dimOp = b.create<memref::DimOp>(loc, input, index);
    dynamicDims.push_back(dimOp);
  }
  auto newAlloc = b.create<memref::AllocOp>(loc, dstMemrefTy, dynamicDims);
  return newAlloc;
}

// creat copy for input Arg
/* Op(A)
=> newA = Alloc() in dstMemrefTy (new space);
   copy(A, newA);
   Op(newA)  // set outside
and record {{A, dstSpace}, newA} in copyPairToCopyTargets
*/
Value createCopyInputArg(Operation *op, Value oldArg, MemRefType dstMemrefTy,
                         Attribute desiredSpaceAttr,
                         DenseMap<CopyType_t, Value> &copyPairToCopyTargets) {
  OpBuilder b(op);
  auto loc = op->getLoc();
  auto newAlloc = createNewAllocWithDstMemrefTy(b, loc, oldArg, dstMemrefTy);
  auto newArg = newAlloc.getResult();
  b.create<memref::CopyOp>(loc, oldArg, newArg);

  CopyType_t copyKey = {oldArg, desiredSpaceAttr};
  copyPairToCopyTargets.try_emplace(copyKey, newArg);
  return newArg;
}

// creat copy for return
/* A = Op()
   someusers(A)
=> A = Op()
   newA = Alloc() in dstMemrefTy (new space);
   copy(A, newA)
   someusers(newA)
and record {{A, dstSpace}, newA} in copyPairToCopyTargets
*/
Value createCopyReturn(Operation *op, Value oldArg, MemRefType dstMemrefTy,
                       DenseMap<CopyType_t, Value> &copyPairToCopyTargets) {
  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  auto loc = op->getLoc();
  auto oriOldArgUsers = oldArg.getUsers();
  auto newAlloc = createNewAllocWithDstMemrefTy(b, loc, oldArg, dstMemrefTy);
  auto newArg = newAlloc.getResult();
  SmallPtrSet<Operation *, 4> excepts;
  // if the shape of oldArg is dynamic, oldArg is used for DimOp when create a
  // new AllocOp. Therefore replacement needs to exclude these operators to
  // avoid circular dependencies
  for (auto curUser : oldArg.getUsers()) {
    bool needExclude = true;
    for (auto oldUser : oriOldArgUsers) {
      if (oldUser == curUser) {
        needExclude = false;
      }
    }
    if (needExclude) {
      excepts.insert(curUser);
    }
  }
  oldArg.replaceAllUsesExcept(newArg, excepts);
  b.create<memref::CopyOp>(loc, oldArg, newArg);
  CopyType_t copyKey = {oldArg, dstMemrefTy.getMemorySpace()};
  copyPairToCopyTargets.try_emplace(copyKey, newArg);
  return newArg;
}

// creat copy for output Arg
/* Op(A)
=> newA = Alloc() in dstMemrefTy (new space);
   Op(newA)  // set outside
   copy(newA, A);
and record {{newA, srcSpace}, A} in copyPairToCopyTargets
*/
Value createCopyOutputArg(Operation *op, Value oldArg, MemRefType dstMemrefTy,
                          DenseMap<CopyType_t, Value> &copyPairToCopyTargets) {
  OpBuilder b(op);
  auto loc = op->getLoc();
  // create alloc before op
  auto newAlloc = b.create<memref::AllocOp>(loc, dstMemrefTy);
  auto newArg = newAlloc.getResult();
  // create copy after op
  b.setInsertionPointAfter(op);
  b.create<memref::CopyOp>(loc, newArg, oldArg);
  auto srcSpaceAttr = oldArg.getType().dyn_cast<MemRefType>().getMemorySpace();
  CopyType_t copyKey = {newArg, srcSpaceAttr};
  copyPairToCopyTargets.try_emplace(copyKey, oldArg);
  return newArg;
}

Value createCopyArg(Operation *op, Value oldArg, MemRefType dstMemrefTy,
                    Attribute desiredSpaceAttr,
                    DenseMap<CopyType_t, Value> &copyPairToCopyTargets,
                    ArgSideEffectType aSETy) {
  if (aSETy == ArgSideEffectType::kInput) {
    return createCopyInputArg(op, oldArg, dstMemrefTy, desiredSpaceAttr,
                              copyPairToCopyTargets);
  }

  if (aSETy == ArgSideEffectType::kOutput) {
    return createCopyOutputArg(op, oldArg, dstMemrefTy, copyPairToCopyTargets);
  }

  return Value();
}

// update function types for args recursively
void updateFuncArgTypes(
    func::FuncOp func, ModuleOp m,
    DenseMap<func::FuncOp, UpdateFuncType_t> &funcToUpdateTypes,
    DenseMap<CopyType_t, Value> &copyPairToCopyTargets, size_t offset,
    Attribute spaceAttr, ArgSideEffectAnalysis *analysis) {
  // skip if suggest spaceAttr is empty
  // or already right space
  if (isEmptyStringAttr(spaceAttr) ||
      isFuncCorrectSpace(func, offset, spaceAttr, true /*isArg*/)) {
    return;
  }

  // initialize funcToUpdateTypes
  if (funcToUpdateTypes.count(func) == 0) {
    FunctionType funcType = func.getFunctionType();
    funcToUpdateTypes.try_emplace(
        func,
        SmallVector<Type, 4>(funcType.getInputs().begin(),
                             funcType.getInputs().end()),
        SmallVector<Type, 4>(funcType.getResults().begin(),
                             funcType.getResults().end()));
  }

  auto &newUpdateTypes = funcToUpdateTypes[func];
  // update argType
  auto &argType = newUpdateTypes.first[offset];

  if (auto MemrefTy = argType.dyn_cast<MemRefType>()) {
    auto newArgType = cloneMemRefTypeWithMemSpace(MemrefTy, spaceAttr);
    argType = newArgType;
  }

  // rewrite body if it's not empty
  if (!func.empty()) {
    Value arg = func.getArgument(offset);
    DenseMap<Attribute, SmallVector<func::FuncOp, 4>> spaceToCalleeFuncs;
    arg.setType(argType);

    // handle users
    llvm::DenseMap<Operation *, size_t> opOrder;
    size_t idx = 0;
    // Get the order of all operations
    func.walk([&](Operation *op) { opOrder[op] = idx++; });
    llvm::SmallVector<Operation *> argUsers = llvm::to_vector(arg.getUsers());
    // Sort the users of this arg according to order
    llvm::sort(argUsers, [&](Operation *lhs, Operation *rhs) {
      return opOrder[lhs] < opOrder[rhs];
    });

    for (auto user : argUsers) {

      // handle call
      if (auto callOp = dyn_cast<CallOp>(user)) {
        auto anotherFunc = m.lookupSymbol<FuncOp>(callOp.getCallee());

        if (anotherFunc == nullptr || !anotherFunc.isPrivate()) {
          continue;
        }

        if (auto privateSpaceAttr =
                anotherFunc->getAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {

          if (privateSpaceAttr != spaceAttr) {
            // check this specific exist or not
            CopyType_t copyKey = {arg, privateSpaceAttr};
            FunctionType privateFuncType = anotherFunc.getFunctionType();
            for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
              if (arg != callOp.getOperand(i)) {
                continue;
              };

              if (copyPairToCopyTargets.count(copyKey) == 0) {
                // if copy not exist, insert copy
                auto argSEType = analysis->getType(user, i);
                auto newArg = createCopyArg(
                    user, arg,
                    privateFuncType.getInput(i).dyn_cast<MemRefType>(),
                    privateSpaceAttr, copyPairToCopyTargets, argSEType);
                callOp.setOperand(i, newArg);
              } else {
                // if copy already exist, directly refer it
                auto taget = copyPairToCopyTargets[copyKey];
                callOp.setOperand(i, taget);
              }
            }
          }
        } else {
          // if not specified device, we assume it supports all devices
          // then perform the same space recurively
          for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
            if (arg != callOp.getOperand(i)) {
              continue;
            }
            updateFuncArgTypes(anotherFunc, m, funcToUpdateTypes,
                               copyPairToCopyTargets, i, spaceAttr, analysis);
          }
        }
      } else {
        // TODO handle regular op
      }
    }
  }
}

// Update function types for return types recursively
void updateFuncReturnTypes(
    FuncOp func, ModuleOp m,
    DenseMap<FuncOp, UpdateFuncType_t> &funcToUpdateTypes,
    DenseMap<CopyType_t, Value> &copyPairToCopyTargets, size_t offset,
    Attribute spaceAttr) {
  // skip if suggest spaceAttr is empty
  // or already right space
  if (isEmptyStringAttr(spaceAttr) ||
      isFuncCorrectSpace(func, offset, spaceAttr, false /*isArg*/)) {
    return;
  }

  // initialize funcToUpdateTypes
  if (funcToUpdateTypes.count(func) == 0) {
    FunctionType funcType = func.getFunctionType();
    funcToUpdateTypes.try_emplace(
        func,
        SmallVector<Type, 4>(funcType.getInputs().begin(),
                             funcType.getInputs().end()),
        SmallVector<Type, 4>(funcType.getResults().begin(),
                             funcType.getResults().end()));
  }

  auto &newUpdateTypes = funcToUpdateTypes[func];
  // update retType
  auto &retType = newUpdateTypes.second[offset];

  if (auto MemrefTy = retType.dyn_cast<MemRefType>()) {
    auto newRetType = cloneMemRefTypeWithMemSpace(MemrefTy, spaceAttr);
    retType = newRetType;
  }

  if (!func.empty()) {
    func::ReturnOp retOp = *func.getOps<func::ReturnOp>().begin();
    Value ret = retOp.getOperand(offset);

    if (auto callOp = ret.getDefiningOp<CallOp>()) {
      // handle return as a call's results
      if (auto anotherFunc = m.lookupSymbol<FuncOp>(callOp.getCallee())) {

        if (isFuncNotCompatiableWithSpace(anotherFunc, spaceAttr)) {
          // insert a CopyFrom after the CallOp
          auto retMemrefTy = retType.dyn_cast<MemRefType>();

          for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
            if (ret != callOp.getResult(i)) {
              continue;
            }

            auto newRet = createCopyReturn(callOp, ret, retMemrefTy,
                                           copyPairToCopyTargets);
            ret = newRet;
            break;
          }

        } else {
          // compatiable function
          ret.setType(retType);
          for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
            if (ret != callOp.getResult(i)) {
              continue;
            }
            updateFuncReturnTypes(anotherFunc, m, funcToUpdateTypes,
                                  copyPairToCopyTargets, i, spaceAttr);
            break;
          }
        }
      }
    } else {
      // regular alloc
      LLVM_DEBUG(llvm::dbgs()
                 << "arg is modified in " << func.getName() << "\n");
    }
  }
}

// update op's types
void updateOpTypes(FuncOp func, ModuleOp m,
                   DenseMap<CopyType_t, Value> &copyPairToCopyTargets,
                   ArgSideEffectAnalysis *analysis) {
  // rewrite all types
  for (auto &block : func.getBlocks()) {
    for (auto &op : block.without_terminator()) {
      if (auto viewLikeOp = llvm::dyn_cast<ViewLikeOpInterface>(op)) {
        auto src = viewLikeOp.getViewSource();
        auto srcType = dyn_cast<MemRefType>(src.getType());
        if (!srcType)
          continue;
        auto srcSpace = srcType.getMemorySpace();
        if (!srcSpace)
          continue;

        auto currSpace = srcSpace;
        // if op has space attribute, use it as memory space
        if (auto opSpaceAttr = op.getAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
          if (srcSpace != opSpaceAttr) {
            // insert copy if src space is different with spaceAttr
            auto newSrcType = cloneMemRefTypeWithMemSpace(srcType, opSpaceAttr);
            auto newArg = createCopyInputArg(&op, src, newSrcType, opSpaceAttr,
                                             copyPairToCopyTargets);
            op.setOperand(0, newArg);
            currSpace = opSpaceAttr;
          }
        }
        // propagate memory space from currSpace to dest
        for (auto result : op.getResults()) {
          auto dstType = result.getType().dyn_cast<MemRefType>();
          if (!dstType)
            continue;
          auto dstSpace = dstType.getMemorySpace();

          if (dstSpace) {
            if (dstSpace != currSpace) {
              // insert copy if dst space was already set to different space
              auto newDstType = cloneMemRefTypeWithMemSpace(dstType, currSpace);
              result.setType(newDstType);
              createCopyReturn(viewLikeOp, result, dstType,
                               copyPairToCopyTargets);
            }
          } else {
            // set to spaceAttr if no space
            auto newDstType = cloneMemRefTypeWithMemSpace(dstType, currSpace);
            result.setType(newDstType);
          }
        }
      } else if (auto opSpaceAttr =
                     op.getAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          auto operand = op.getOperand(i);
          if (auto MemrefTy = operand.getType().dyn_cast<MemRefType>()) {
            auto curSpace = MemrefTy.getMemorySpace();

            if (curSpace == nullptr) {
              // if no space, use opSpaceAttr
              auto newOperandType =
                  cloneMemRefTypeWithMemSpace(MemrefTy, opSpaceAttr);
              operand.setType(newOperandType);
            } else if (opSpaceAttr != curSpace) {
              // insert copy when curSpace is not desired opSpaceAttr
              CopyType_t copyKey = {operand, opSpaceAttr};

              if (copyPairToCopyTargets.count(copyKey) == 0) {
                // if copy not exist, insert copy
                auto argSEType = analysis->getType(&op, i);
                auto newArg = createCopyArg(&op, operand, MemrefTy, opSpaceAttr,
                                            copyPairToCopyTargets, argSEType);
                op.setOperand(i, newArg);
              } else {
                // if copy already exist, directly refer it
                auto taget = copyPairToCopyTargets[copyKey];
                op.setOperand(i, taget);
              }
            } // if else
          }   // if MemrefTy
        }     // for i < op.getNumOperands()

        // set operand type
        for (auto operand : op.getOperands()) {
          if (auto MemrefTy = operand.getType().dyn_cast<MemRefType>()) {
            auto newOperandType =
                cloneMemRefTypeWithMemSpace(MemrefTy, opSpaceAttr);
            operand.setType(newOperandType);
          }
        }

        // set result type in case it has
        for (auto result : op.getResults()) {
          if (auto MemrefTy = result.getType().dyn_cast<MemRefType>()) {
            auto newOperandType =
                cloneMemRefTypeWithMemSpace(MemrefTy, opSpaceAttr);
            result.setType(newOperandType);
          }
        }
      }
    } // for op in block.without_terminator()
  }

  // respect to function return type
  for (auto &&retOp : func.getOps<ReturnOp>()) {
    for (auto &&opOperand : retOp->getOpOperands()) {
      auto operandType = opOperand.get().getType().dyn_cast<MemRefType>();
      auto resultType = func.getFunctionType()
                            .getResult(opOperand.getOperandNumber())
                            .dyn_cast<MemRefType>();
      if (!resultType || !operandType)
        continue;

      auto operandSpace = operandType.getMemorySpace();
      auto resultSpace = resultType.getMemorySpace();

      if (operandSpace == resultSpace)
        continue;

      CopyType_t copyKey = {opOperand.get(), resultSpace};
      if (copyPairToCopyTargets.count(copyKey) == 0) {
        auto newArg = createCopyInputArg(retOp, opOperand.get(), resultType,
                                         resultSpace, copyPairToCopyTargets);
        opOperand.set(newArg);
      } else {
        auto taget = copyPairToCopyTargets[copyKey];
        opOperand.set(taget);
      }
    }
  }
}

// set op within a funcOp f to space
void setOpSpace(FuncOp f, const std::string &space) {
  if (f.empty()) {
    return;
  }

  auto m = f->getParentOfType<ModuleOp>();
  for (auto &block : f.getBlocks()) {
    for (auto &op : block.without_terminator()) {
      // skip if attr was set
      if (op.hasAttr(SPACE_ATTR_NAME) || isMemref(op)) {
        continue;
      }

      if (auto callOp = dyn_cast<CallOp>(op)) {
        // handle a call
        auto anotherFunc = m.lookupSymbol<FuncOp>(callOp.getCallee());
        if (anotherFunc == nullptr || anotherFunc->hasAttr(SPACE_ATTR_NAME)) {
          continue;
        }
        anotherFunc->setAttr(SPACE_ATTR_NAME, getOrCreateSpaceAttr(m, space));
        // recursive
        setOpSpace(anotherFunc, space);
      } else {
        // handle a regular op
        op.setAttr(SPACE_ATTR_NAME, getOrCreateSpaceAttr(m, space));
      }
    }
  }
}

std::string deduceSpace(ModuleOp m, Value value, std::string fallbackSpace);

std::string deduceFuncArgSpace(ModuleOp m, FuncOp func, size_t arg_idx,
                               std::string fallbackSpace) {
  // respect to function attribute first
  if (auto spaceAttr = func->getAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
    auto space = spaceAttr.getValue().str();
    if (!space.empty()) {
      return space;
    }
  }

  // deduce recursively for non-external function
  if (!func.isExternal()) {
    return deduceSpace(m, func.getArgument(arg_idx), fallbackSpace);
  }

  // fallback
  return fallbackSpace;
}

std::string deduceFuncResultSpace(ModuleOp m, FuncOp func, size_t result_idx,
                                  std::string fallbackSpace) {
  // respect to function attribute first
  if (auto spaceAttr = func->getAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
    auto space = spaceAttr.getValue().str();
    if (!space.empty()) {
      return space;
    }
  }

  // deduce recursively for non-external function
  if (!func.isExternal()) {
    if (auto terminator = func.getBody().front().getTerminator()) {
      return deduceSpace(m, terminator->getOperand(result_idx), fallbackSpace);
    }
  }

  // fallback
  return fallbackSpace;
}

std::string deduceSpace(ModuleOp m, Value value, std::string fallbackSpace) {
  // deduce from defining op
  {
    std::string deduceSpace;
    if (auto definingOp = value.getDefiningOp()) {
      if (auto callOp = dyn_cast<CallOp>(definingOp)) {
        if (auto callee = m.lookupSymbol<FuncOp>(callOp.getCallee())) {
          deduceSpace = deduceFuncResultSpace(
              m, callee, cast<OpResult>(value).getResultNumber(),
              fallbackSpace);
        }
      } else if (auto curSpaceAttr =
                     definingOp->getAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
        deduceSpace = curSpaceAttr.getValue().str();
      }
    }
    if (!deduceSpace.empty())
      return deduceSpace;
  }

  // deduce from uses
  for (auto &&operand : value.getUses()) {
    std::string deduceSpace;
    if (auto callOp = dyn_cast<CallOp>(operand.getOwner())) {
      if (auto callee = m.lookupSymbol<FuncOp>(callOp.getCallee())) {
        deduceSpace = deduceFuncArgSpace(m, callee, operand.getOperandNumber(),
                                         fallbackSpace);
      }
    } else if (auto curSpaceAttr =
                   operand.getOwner()->getAttrOfType<StringAttr>(
                       SPACE_ATTR_NAME)) {
      deduceSpace = curSpaceAttr.getValue().str();
    }

    if (!deduceSpace.empty())
      return deduceSpace;
  }

  // fallback
  return fallbackSpace;
}

struct SetAllSpacePass : public SetAllSpaceBase<SetAllSpacePass> {
  explicit SetAllSpacePass() = default;

  SetAllSpacePass(const std::string &entryFuncName, const std::string &space_,
                  ArgSideEffectAnalysis *externalAnalysis = nullptr)
      : SetAllSpaceBase() {
    entryFunc = entryFuncName;
    space = space_;
    if (nullptr != externalAnalysis) {
      analysis = externalAnalysis;
    } else {
      interalAnalysis = new ArgSideEffectAnalysis;
      analysis = interalAnalysis;
    }
  }

  ~SetAllSpacePass() {
    if (nullptr != interalAnalysis) {
      delete interalAnalysis;
    }
  }

  void runOnOperation() override {
    // early termination
    if (entryFunc.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied function.\n");
      return;
    }

    ModuleOp m = getOperation();
    FuncOp funcOp = m.lookupSymbol<FuncOp>(entryFunc);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find the speficied function "
                              << entryFunc.getValue() << "\n");
      return;
    }

    auto ctx = m->getContext();
    auto newSpace = StringAttr::get(ctx, space);

    DenseMap<FuncOp, UpdateFuncType_t> funcToArgTypes;
    DenseMap<CopyType_t, Value> copyPairToCopyTargets;

    // resolve entry function
    // argumenets
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      updateFuncArgTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                         newSpace, analysis);
    }

    // results
    for (unsigned i = 0, e = funcOp.getNumResults(); i < e; ++i) {
      updateFuncReturnTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                            newSpace);
    }

    // local alloc
    for (auto alloc : funcOp.getOps<memref::AllocOp>()) {
      auto ret = alloc.getResult();
      if (auto MemrefTy = ret.getType().dyn_cast<MemRefType>()) {
        auto newRetType = cloneMemRefTypeWithMemSpace(MemrefTy, newSpace);
        ret.setType(newRetType);
      }
    }

    // rewrite FunctionType
    for (auto it : funcToArgTypes) {
      it.first.setType(
          FunctionType::get(ctx, it.second.first, it.second.second));
    }
  }

  ArgSideEffectAnalysis *interalAnalysis = nullptr;
  ArgSideEffectAnalysis *analysis = nullptr;
};

struct SetArgSpacePass : public SetArgSpaceBase<SetArgSpacePass> {
  SetArgSpacePass(const std::string &entryFuncName, const std::string &space,
                  bool allowOutWritable, bool deduceSpace,
                  ArgSideEffectAnalysis *externalAnalysis = nullptr)
      : SetArgSpaceBase() {
    entryFunc = entryFuncName;
    allSpace = space;
    allowArgWritable = allowOutWritable;
    autoDeduce = deduceSpace;

    if (nullptr != externalAnalysis) {
      analysis = externalAnalysis;
    } else {
      interalAnalysis = new ArgSideEffectAnalysis;
      analysis = interalAnalysis;
    }
  }

  SetArgSpacePass(const std::string &entryFuncName,
                  ArrayRef<std::string> argList, ArrayRef<std::string> retList,
                  bool allowOutWritable,
                  ArgSideEffectAnalysis *externalAnalysis = nullptr)
      : SetArgSpaceBase(), argSpaces(argList.begin(), argList.end()),
        retSpaces(retList.begin(), retList.end()) {
    entryFunc = entryFuncName;
    allowArgWritable = allowOutWritable;

    if (nullptr != externalAnalysis) {
      analysis = externalAnalysis;
    } else {
      interalAnalysis = new ArgSideEffectAnalysis;
      analysis = interalAnalysis;
    }
  }

  ~SetArgSpacePass() {
    if (nullptr != interalAnalysis) {
      delete interalAnalysis;
    }
  }

  void runOnOperation() override {
    // TODO: after supporting allowArgWritable version, remove this.
    if (allowArgWritable) {
      LLVM_DEBUG(llvm::dbgs()
                 << "allowArgWritable version is not implmented yet.\n");
    }

    // early termination
    if (entryFunc.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied function.\n");
      return;
    }

    ModuleOp m = getOperation();
    auto ctx = m.getContext();
    FuncOp funcOp = m.lookupSymbol<FuncOp>(entryFunc);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find the speficied function "
                              << entryFunc.getValue() << "\n");
      return;
    }

    // parse spaces
    if (argSpaces.empty()) {
      if (autoDeduce.getValue()) {
        for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
          argSpaces.push_back(
              deduceFuncArgSpace(m, funcOp, i, allSpace.getValue()));
        }
      } else if (!allSpace.getValue().empty()) {
        argSpaces.push_back(allSpace.getValue());
      }
    }

    if (argSpaces.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied argSpaces.\n");
      return;
    }

    if (retSpaces.empty()) {
      if (autoDeduce.getValue()) {
        for (unsigned i = 0, e = funcOp.getNumResults(); i < e; ++i) {
          retSpaces.push_back(
              deduceFuncResultSpace(m, funcOp, i, allSpace.getValue()));
        }
      } else if (!allSpace.getValue().empty()) {
        retSpaces.push_back(allSpace.getValue());
      }
    }

    if (retSpaces.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied return Spaces.\n");
      return;
    }

    DenseMap<FuncOp, UpdateFuncType_t> funcToArgTypes;
    DenseMap<CopyType_t, Value> copyPairToCopyTargets;

    // resolve entry function
    // argumenets
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      auto newSpace = getOrCreateSpaceAttr(m, getSpace(argSpaces, i));
      updateFuncArgTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                         newSpace, analysis);
    }
    // results
    for (unsigned i = 0, e = funcOp.getNumResults(); i < e; ++i) {
      auto newSpace = getOrCreateSpaceAttr(m, getSpace(retSpaces, i));
      updateFuncReturnTypes(funcOp, m, funcToArgTypes, copyPairToCopyTargets, i,
                            newSpace);
    }

    // resolve device functions for ops
    // Note, it will also set a blank func (func without specifying device)
    // called by a device func into a device
    /*
     *  func A
     *  func B attribute {device = test} {
     *    call A;
     *  }
     * ==>
     *  func A attribute {device = test}
     *  func B attribute {device = test} {
     *    call A;
     *  }
     */
    for (auto deviceFuncOp : m.getOps<FuncOp>()) {
      // skip non-private or one without device attr
      if (!deviceFuncOp.isPrivate() ||
          !deviceFuncOp->hasAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
        continue;
      }

      auto deviceAttr =
          deviceFuncOp->getAttrOfType<StringAttr>(SPACE_ATTR_NAME);

      setOpSpace(deviceFuncOp, deviceAttr.str());
    }

    // resolve device functions
    for (auto deviceFuncOp : m.getOps<FuncOp>()) {
      // skip non-private or one without device attr
      if (!deviceFuncOp.isPrivate() ||
          !deviceFuncOp->hasAttrOfType<StringAttr>(SPACE_ATTR_NAME)) {
        continue;
      }

      auto deviceAttr =
          deviceFuncOp->getAttrOfType<StringAttr>(SPACE_ATTR_NAME);

      // argumenets
      for (unsigned i = 0, e = deviceFuncOp.getNumArguments(); i < e; ++i) {
        updateFuncArgTypes(deviceFuncOp, m, funcToArgTypes,
                           copyPairToCopyTargets, i, deviceAttr, analysis);
      }

      // results
      for (unsigned i = 0, e = deviceFuncOp.getNumResults(); i < e; ++i) {
        updateFuncReturnTypes(deviceFuncOp, m, funcToArgTypes,
                              copyPairToCopyTargets, i, deviceAttr);
      }

      // handle callee
      auto &newArgTypes = funcToArgTypes[deviceFuncOp];
      auto maybeSymbolUses = deviceFuncOp.getSymbolUses(m);
      for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
        if (auto callOp = dyn_cast<CallOp>(symbolUse.getUser())) {
          // arguement
          for (unsigned i = 0, e = callOp.getNumOperands(); i < e; ++i) {
            auto operand = callOp.getOperand(i);
            if (auto memrefType = operand.getType().dyn_cast<MemRefType>()) {
              auto curSpace = memrefType.getMemorySpace();
              // insert arg copy if operand was already set with different space
              // or operand is the result of some function call with
              // incompatible space
              bool needCopy = curSpace && deviceAttr != curSpace;
              if (auto anotherCall = operand.getDefiningOp<CallOp>())
                if (auto anotherFunc =
                        m.lookupSymbol<FuncOp>(anotherCall.getCallee()))
                  needCopy |=
                      isFuncNotCompatiableWithSpace(anotherFunc, deviceAttr);
              if (needCopy) {
                CopyType_t copyKey = {operand, deviceAttr};

                if (copyPairToCopyTargets.count(copyKey) == 0) {
                  // if copy not exist, insert copy
                  auto argSEType = analysis->getType(callOp, i);
                  auto newArg =
                      createCopyArg(callOp, operand, memrefType, deviceAttr,
                                    copyPairToCopyTargets, argSEType);
                  callOp->setOperand(i, newArg);
                } else {
                  // if copy already exist, directly refer it
                  auto taget = copyPairToCopyTargets[copyKey];
                  callOp->setOperand(i, taget);
                }
              }
            }
            callOp.getOperand(i).setType(newArgTypes.first[i]);
          }

          // results
          for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i) {
            callOp.getResult(i).setType(newArgTypes.second[i]);
          }
        }
      }
    }

    // rewrite FunctionType
    for (auto it : funcToArgTypes) {
      it.first.setType(
          FunctionType::get(ctx, it.second.first, it.second.second));
      // update all func ops
      updateOpTypes(it.first, m, copyPairToCopyTargets, analysis);
    };

    // handle global op
    DenseMap<GlobalType_t, memref::GlobalOp> globalPairToGlobalTargets;
    SymbolTable symbolTable(m);
    auto globals = llvm::to_vector(m.getOps<memref::GlobalOp>());
    for (auto globalOp : globals) {
      auto maybeSymbolUses = globalOp.getSymbolUses(m);
      for (SymbolTable::SymbolUse symbolUse : *maybeSymbolUses) {
        auto symbolUser = dyn_cast<memref::GetGlobalOp>(symbolUse.getUser());
        if (!symbolUser)
          continue;

        auto globalSpace = globalOp.getType().getMemorySpace();
        auto useSpace = symbolUser.getType().getMemorySpace();

        if (globalSpace == useSpace)
          continue;

        GlobalType_t key{globalOp, useSpace};
        auto &&iter = globalPairToGlobalTargets.find(key);
        if (iter == globalPairToGlobalTargets.end()) {
          auto newGlobalOp = cast<memref::GlobalOp>(globalOp->clone());
          // set new type with space
          auto newMemRefType =
              cloneMemRefTypeWithMemSpace(globalOp.getType(), useSpace);
          newGlobalOp.setType(newMemRefType);

          // append space suffix to sym name
          auto newGlobalName = globalOp.getSymName().str();
          if (auto useSpaceStrAttr = useSpace.dyn_cast_or_null<StringAttr>()) {
            newGlobalName += "_" + useSpaceStrAttr.getValue().str();
          }
          newGlobalOp.setSymName(newGlobalName);

          // insert into symbol table
          symbolTable.insert(newGlobalOp);

          symbolUser.setName(newGlobalName);
          globalPairToGlobalTargets[key] = newGlobalOp;
        } else {
          symbolUser.setName(iter->second.getSymName());
        }
      }

      if (SymbolTable::symbolKnownUseEmpty(globalOp, m)) {
        symbolTable.erase(globalOp);
      }
    }
  }

  llvm::SmallVector<std::string, 16> argSpaces;
  llvm::SmallVector<std::string, 16> retSpaces;
  ArgSideEffectAnalysis *interalAnalysis = nullptr;
  ArgSideEffectAnalysis *analysis = nullptr;
};

struct SetOpSpacePass : public SetOpSpaceBase<SetOpSpacePass> {

  SetOpSpacePass(const std::string &entryFuncName, const std::string &spaceName)
      : SetOpSpaceBase() {
    entryFunc = entryFuncName;
    space = spaceName;
  }

  void runOnOperation() override {
    // early termination
    if (entryFunc.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No speficied function.\n");
      return;
    }

    FuncOp f = getOperation();
    // early termination
    if (f.getName() != entryFunc) {
      return;
    }

    setOpSpace(f, space);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetAllSpacePass(const std::string &entryFunc,
                            const std::string &space,
                            byteir::ArgSideEffectAnalysis *analysis) {
  return std::make_unique<SetAllSpacePass>(entryFunc, space, analysis);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetArgSpacePass(const std::string &entryFunc,
                            const std::string &allSpace, bool allowArgWritable,
                            bool autoDeduce,
                            byteir::ArgSideEffectAnalysis *analysis) {
  return std::make_unique<SetArgSpacePass>(
      entryFunc, allSpace, allowArgWritable, autoDeduce, analysis);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createSetArgSpacePass(
    const std::string &entryFunc, llvm::ArrayRef<std::string> argSpaces,
    llvm::ArrayRef<std::string> retSpaces, bool allowArgWritable,
    byteir::ArgSideEffectAnalysis *analysis) {
  return std::make_unique<SetArgSpacePass>(entryFunc, argSpaces, retSpaces,
                                           allowArgWritable, analysis);
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createSetOpSpacePass(const std::string &entryFunc,
                           const std::string &space) {
  return std::make_unique<SetOpSpacePass>(entryFunc, space);
}
