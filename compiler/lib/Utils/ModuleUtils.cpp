//===- ModuleUtils.cpp ----------------------------------------------------===//
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

#include "byteir/Utils/ModuleUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"

#include <string>
#include <unordered_map>

using namespace mlir;

namespace {

const char *module0Name = "__byteir__merge_model_0";
const char *module1Name = "__byteir__merge_model_1";

func::FuncOp renameAndCloneFuncToNewModule(ModuleOp m, func::FuncOp func,
                                           const std::string &newFuncName) {
  OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
  auto newFunc = builder.create<func::FuncOp>(func->getLoc(), newFuncName,
                                              func.getFunctionType());
  newFunc.setPrivate();
  IRMapping emptyBvm;
  func.cloneInto(newFunc, emptyBvm);
  return newFunc;
}

ModuleOp mergeTwoModulesByName(ModuleOp module0, ModuleOp module1,
                               MLIRContext *context) {
  func::FuncOp func0 = *module0.getOps<func::FuncOp>().begin();
  func::FuncOp func1 = *module1.getOps<func::FuncOp>().begin();
  if (func0.getNumResults() != func1.getNumArguments()) {
    return nullptr;
  }
  if (func0.getNumResults() == 0) {
    return nullptr;
  }

  // get inputs and outputs name from byteir.entry_point
  auto func0EntryPointDict =
      func0->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
  auto func1EntryPointDict =
      func1->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
  assert(func0EntryPointDict && func1EntryPointDict &&
         "byteir.entry_point should be dict attr.");
  auto func0OutputNamesAttr =
      cast<mlir::ArrayAttr>(func0EntryPointDict.get("outputs"));
  auto func1InputNamesAttr =
      cast<mlir::ArrayAttr>(func1EntryPointDict.get("inputs"));
  SmallVector<std::string> func0OutputNames =
      llvm::to_vector(llvm::map_range(func0OutputNamesAttr, [&](Attribute i) {
        return cast<StringAttr>(i).getValue().str();
      }));
  SmallVector<std::string> func1InputNames =
      llvm::to_vector(llvm::map_range(func1InputNamesAttr, [&](Attribute i) {
        return cast<StringAttr>(i).getValue().str();
      }));

  // get map of func1's inputs name to func0's index
  std::unordered_map<std::string, size_t> func1NameToFunc0Index;
  for (const auto &name : func1InputNames) {
    auto findNames = [&](const std::string &name) -> bool {
      for (auto it : llvm::enumerate(func0OutputNames)) {
        if (name == it.value()) {
          func1NameToFunc0Index[name] = it.index();
          return true;
        }
      }
      return false;
    };
    if (!findNames(name)) {
      return nullptr;
    }
  }

  // check types
  for (size_t i = 0; i < func1.getNumArguments(); i++) {
    if (func1.getArgumentTypes()[i] !=
        func0.getResultTypes()[func1NameToFunc0Index[func1InputNames[i]]]) {
      return nullptr;
    }
  }

  // create new module, clone func0 and func1 to new module
  ModuleOp m = ModuleOp::create(UnknownLoc::get(context));
  func::FuncOp newFunc0 = renameAndCloneFuncToNewModule(m, func0, module0Name);
  func::FuncOp newFunc1 = renameAndCloneFuncToNewModule(m, func1, module1Name);

  // create main function in new module
  OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
  auto mainFunc = builder.create<func::FuncOp>(
      UnknownLoc::get(context), "main",
      FunctionType::get(context, func0.getArgumentTypes(),
                        func1.getResultTypes()));
  Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto callOp0 = builder.create<func::CallOp>(
      UnknownLoc::get(context), newFunc0, mainFunc.getArguments());
  SmallVector<Value> callOp1Operands;
  for (const auto &name : func1InputNames) {
    callOp1Operands.push_back(
        callOp0.getResults()[func1NameToFunc0Index[name]]);
  }
  auto callOp1 = builder.create<func::CallOp>(UnknownLoc::get(context),
                                              newFunc1, callOp1Operands);
  builder.create<func::ReturnOp>(UnknownLoc::get(context),
                                 callOp1.getResults());

  // set new "byteir.entry_point" attr on main function
  newFunc0->removeAttr(getByteIREntryPointName());
  newFunc1->removeAttr(getByteIREntryPointName());
  NamedAttribute newInputsAttr =
      NamedAttribute(builder.getStringAttr("inputs"),
                     cast<ArrayAttr>(func0EntryPointDict.get("inputs")));
  NamedAttribute newOutputsAttr =
      NamedAttribute(builder.getStringAttr("outputs"),
                     cast<ArrayAttr>(func1EntryPointDict.get("outputs")));
  mainFunc->setAttr(
      getByteIREntryPointName(),
      DictionaryAttr::get(context, {newInputsAttr, newOutputsAttr}));
  return m;
}

ModuleOp mergeTwoModulesByOrder(ModuleOp module0, ModuleOp module1,
                                MLIRContext *context) {
  func::FuncOp func0 = *module0.getOps<func::FuncOp>().begin();
  func::FuncOp func1 = *module1.getOps<func::FuncOp>().begin();
  if (func0.getNumResults() != func1.getNumArguments()) {
    return nullptr;
  }
  if (func0.getNumResults() == 0) {
    return nullptr;
  }
  // check types
  for (size_t i = 0; i < func1.getNumArguments(); i++) {
    // func0 and func1 should have the same context
    if (func0.getResultTypes()[i] != func1.getArgumentTypes()[i]) {
      return nullptr;
    }
  }

  // create new module, clone func0 and func1 to new module
  ModuleOp m = ModuleOp::create(UnknownLoc::get(context));
  func::FuncOp newFunc0 = renameAndCloneFuncToNewModule(m, func0, module0Name);
  func::FuncOp newFunc1 = renameAndCloneFuncToNewModule(m, func1, module1Name);

  // create main function in new module
  OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
  auto mainFunc = builder.create<func::FuncOp>(
      UnknownLoc::get(context), "main",
      FunctionType::get(context, func0.getArgumentTypes(),
                        func1.getResultTypes()));
  Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto callOp0 = builder.create<func::CallOp>(
      UnknownLoc::get(context), newFunc0, mainFunc.getArguments());
  auto callOp1 = builder.create<func::CallOp>(UnknownLoc::get(context),
                                              newFunc1, callOp0.getResults());
  builder.create<func::ReturnOp>(UnknownLoc::get(context),
                                 callOp1.getResults());
  return m;
}

} // namespace

OwningOpRef<ModuleOp> mlir::mergeTwoModulesByNameOrOrder(ModuleOp module0,
                                                         ModuleOp module1) {
  assert(module0.getContext() == module1.getContext() &&
         "module0 and module1 should have same context");
  MLIRContext *context = module0.getContext();

  // only support module with one function
  if (llvm::count_if(module0.getOps<func::FuncOp>(),
                     [](func::FuncOp func) { return true; }) != 1) {
    return nullptr;
  }
  if (llvm::count_if(module1.getOps<func::FuncOp>(),
                     [](func::FuncOp func) { return true; }) != 1) {
    return nullptr;
  }

  unsigned module0FuncCountWithEntryPoint =
      llvm::count_if(module0.getOps<func::FuncOp>(), [](func::FuncOp func) {
        return func->hasAttr(getByteIREntryPointName());
      });
  unsigned module1FuncCountWithEntryPoint =
      llvm::count_if(module1.getOps<func::FuncOp>(), [](func::FuncOp func) {
        return func->hasAttr(getByteIREntryPointName());
      });
  if (module0FuncCountWithEntryPoint == 1 &&
      module1FuncCountWithEntryPoint == 1) {
    return mergeTwoModulesByName(module0, module1, context);
  } else if (module0FuncCountWithEntryPoint == 0 &&
             module1FuncCountWithEntryPoint == 0) {
    return mergeTwoModulesByOrder(module0, module1, context);
  }
  return nullptr;
}
