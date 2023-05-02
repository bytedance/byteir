//===- FunctionSupport.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/Common/FunctionSupport.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace function_interface_impl {
// llvm upstream insertFunctionArgument cannot handle inserting multiple args

static inline void
insertFunctionArgumentsEx(FunctionOpInterface op, ArrayRef<unsigned> argIndices,
                          TypeRange argTypes, ArrayRef<DictionaryAttr> argAttrs,
                          ArrayRef<Location> argLocs, unsigned originalNumArgs,
                          Type newType) {
  assert(argIndices.size() == argTypes.size());
  assert(argIndices.size() == argAttrs.size() || argAttrs.empty());
  assert(argIndices.size() == argLocs.size() || argLocs.empty());
  if (argIndices.empty())
    return;

  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();

  // Update the argument attributes of the function.
  ArrayAttr oldArgAttrs = op.getArgAttrsAttr();
  if (oldArgAttrs || !argAttrs.empty()) {
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    newArgAttrs.reserve(originalNumArgs + argIndices.size());
    unsigned oldIdx = 0;
    auto migrate = [&](unsigned oldNum) {
      if (!oldArgAttrs) {
        newArgAttrs.resize(newArgAttrs.size() + oldNum);
      } else {
        auto oldArgAttrRange = oldArgAttrs.getAsRange<DictionaryAttr>();
        auto newIdx = oldIdx + oldNum;
        newArgAttrs.append(oldArgAttrRange.begin() + oldIdx,
                           oldArgAttrRange.begin() + newIdx);
        oldIdx = newIdx;
      }
    };
    for (unsigned i = 0, e = argIndices.size(); i < e; ++i) {
      migrate(i ? argIndices[i] - argIndices[i - 1] - 1 : argIndices[i]);
      newArgAttrs.push_back(argAttrs.empty() ? DictionaryAttr{} : argAttrs[i]);
    }
    migrate(originalNumArgs - oldIdx);
    setAllArgAttrDicts(op, newArgAttrs);
  }

  // Update the function type and any entry block arguments.
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
    entry.insertArgument(argIndices[i], argTypes[i],
                         argLocs.empty() ? op->getLoc() : argLocs[i]);
}
} // namespace function_interface_impl
} // namespace mlir

//
// LWC NOTE This implementation DO NOT support inout,
// meaning directly returning an input as an results
// LWC NOTE Also DO NOT support duplicated results.
//
namespace {

static inline void replicateFuncOpResultSigature(func::FuncOp funcOp) {
  mlir::FunctionType oldFuncType = funcOp.getFunctionType();

  llvm::SmallVector<Type, 16> newInputTypes(oldFuncType.getInputs().begin(),
                                            oldFuncType.getInputs().end());
  newInputTypes.append(oldFuncType.getResults().begin(),
                       oldFuncType.getResults().end());

  unsigned origianlSize = oldFuncType.getInputs().size();
  llvm::SmallVector<unsigned, 16> relocatedIndices;
  relocatedIndices.reserve(oldFuncType.getNumResults());
  for (unsigned int i = 0; i < oldFuncType.getNumResults(); ++i) {
    relocatedIndices.push_back(origianlSize + i);
  }

  mlir::OpBuilder opBuilder(funcOp);

  mlir::FunctionType newFuncType =
      opBuilder.getFunctionType(newInputTypes, {} /*results*/);

  llvm::SmallVector<DictionaryAttr, 4> resultAttrs;
  for (size_t i = 0; i < funcOp.getNumResults(); ++i) {
    NamedAttrList attrList = funcOp.getResultAttrs(i);
    resultAttrs.push_back(attrList.getDictionary(funcOp->getContext()));
  }

  auto funcInterface = cast<FunctionOpInterface>(funcOp.getOperation());
  funcInterface.removeResAttrsAttr();

  mlir::function_interface_impl::insertFunctionArgumentsEx(
      funcInterface, relocatedIndices, oldFuncType.getResults(), resultAttrs,
      {}, origianlSize, newFuncType);
  //  Note the FuncOp's member function seems buggy. Use the above
  //  instread. funcOp.insertArguments(relocatedIndices,
  //  oldFuncType.getResults(), {},
  //  {});
}
} // namespace

void mlir::replicateFuncOpResults(func::FuncOp funcOp) {
  unsigned idx = funcOp.getNumArguments();

  replicateFuncOpResultSigature(funcOp);

  if (funcOp.empty()) {
    return;
  }

  func::ReturnOp retOp =
      cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

  llvm::SmallPtrSet<mlir::Operation *, 16> retAllocs;

  for (auto retVal : retOp.getOperands()) {
    retVal.replaceAllUsesExcept(funcOp.getArgument(idx++), retOp);
    if (auto allocOp = dyn_cast<memref::AllocOp>(retVal.getDefiningOp())) {
      if (retAllocs.count(allocOp) == 0) {
        retAllocs.insert(allocOp);
      }
    }
  }

  mlir::OpBuilder opBuilder(retOp);
  opBuilder.create<func::ReturnOp>(retOp.getLoc());
  retOp.erase();
  for (auto op : retAllocs) {
    op->erase();
  }
}

void mlir::replicateFuncOpResults(
    func::FuncOp funcOp, std::function<void(func::ReturnOp)> retOpHandling) {
  replicateFuncOpResultSigature(funcOp);

  if (funcOp.empty()) {
    return;
  }

  func::ReturnOp retOp =
      cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
  retOpHandling(retOp);
}
