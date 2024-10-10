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
#include <unordered_set>

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

llvm::SmallVector<Type>
getMainFuncArgumentTypes(func::FuncOp func0, func::FuncOp func1,
                         llvm::ArrayRef<int64_t> mapping) {
  std::unordered_set<int64_t> mappingSet(mapping.begin(), mapping.end());
  llvm::SmallVector<Type> mainArgumentTypes(func0.getArgumentTypes());
  for (size_t i = 0; i < func1.getNumArguments(); i++) {
    if (mappingSet.find(i) == mappingSet.end()) {
      mainArgumentTypes.push_back(func1.getArgumentTypes()[i]);
    }
  }
  return mainArgumentTypes;
}

llvm::SmallVector<mlir::Attribute>
getMainFuncInputNames(func::FuncOp func0, func::FuncOp func1,
                      llvm::ArrayRef<int64_t> mapping) {
  std::unordered_set<int64_t> mappingSet(mapping.begin(), mapping.end());
  llvm::SmallVector<mlir::Attribute> inputNames;
  auto func0EntryPointDict =
      func0->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
  auto func1EntryPointDict =
      func1->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
  auto func0InputNamesAttr =
      cast<mlir::ArrayAttr>(func0EntryPointDict.get("inputs"));
  auto func1InputNamesAttr =
      cast<mlir::ArrayAttr>(func1EntryPointDict.get("inputs"));
  inputNames.insert(inputNames.begin(), func0InputNamesAttr.begin(),
                    func0InputNamesAttr.end());
  for (size_t i = 0; i < func1.getNumArguments(); i++) {
    if (mappingSet.find(i) == mappingSet.end()) {
      inputNames.push_back(func1InputNamesAttr[i]);
    }
  }
  return inputNames;
}

llvm::SmallVector<Type>
getMainFuncResultTypes(func::FuncOp func0, func::FuncOp func1,
                       llvm::ArrayRef<int64_t> mapping) {
  llvm::SmallVector<Type> mainResultTypes;
  for (size_t i = 0; i < mapping.size(); i++) {
    if (mapping[i] < 0)
      mainResultTypes.push_back(func0.getResultTypes()[i]);
  }
  mainResultTypes.insert(mainResultTypes.end(), func1.getResultTypes().begin(),
                         func1.getResultTypes().end());
  return mainResultTypes;
}

llvm::SmallVector<mlir::Attribute>
getMainFuncOutputNames(func::FuncOp func0, func::FuncOp func1,
                       llvm::ArrayRef<int64_t> mapping) {
  llvm::SmallVector<mlir::Attribute> outputNames;
  auto func0EntryPointDict =
      func0->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
  auto func1EntryPointDict =
      func1->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
  auto func0OutputNamesAttr =
      cast<mlir::ArrayAttr>(func0EntryPointDict.get("outputs"));
  auto func1OutputNamesAttr =
      cast<mlir::ArrayAttr>(func1EntryPointDict.get("outputs"));
  for (size_t i = 0; i < mapping.size(); i++) {
    if (mapping[i] < 0)
      outputNames.push_back(func0OutputNamesAttr[i]);
  }
  outputNames.insert(outputNames.end(), func1OutputNamesAttr.begin(),
                     func1OutputNamesAttr.end());
  return outputNames;
}

ModuleOp mergeTwoModules(ModuleOp module0, ModuleOp module1,
                         MLIRContext *context, llvm::ArrayRef<int64_t> mapping,
                         bool hasEntryPoint) {
  func::FuncOp func0 = *module0.getOps<func::FuncOp>().begin();
  func::FuncOp func1 = *module1.getOps<func::FuncOp>().begin();
  if (func0.getNumResults() == 0 || func0.getNumResults() != mapping.size()) {
    llvm::errs() << "mapping length must be same as func0's outputs length\n";
    return nullptr;
  }
  // check types
  for (size_t i = 0; i < mapping.size(); i++) {
    if (mapping[i] < 0)
      continue;
    if (func0.getResultTypes()[i] != func1.getArgumentTypes()[mapping[i]]) {
      llvm::errs() << "src and dst tensor of mapping must be same type\n";
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
      FunctionType::get(context,
                        getMainFuncArgumentTypes(func0, func1, mapping),
                        getMainFuncResultTypes(func0, func1, mapping)));
  Block *entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto callOp0 = builder.create<func::CallOp>(
      UnknownLoc::get(context), newFunc0,
      llvm::SmallVector<Value>(mainFunc.getArguments().begin(),
                               mainFunc.getArguments().begin() +
                                   func0.getNumArguments()));
  // collect call func1's operands
  llvm::SmallVector<Value> callOp1Operands(func1.getNumArguments(), nullptr);
  for (size_t i = 0; i < mapping.size(); i++) {
    if (mapping[i] < 0)
      continue;
    callOp1Operands[mapping[i]] = callOp0.getResults()[i];
  }
  auto iter = mainFunc.getArguments().begin() + func0.getNumArguments();
  for (size_t i = 0; i < callOp1Operands.size(); i++) {
    if (callOp1Operands[i] == nullptr) {
      callOp1Operands[i] = *iter;
      iter++;
    }
  }
  auto callOp1 = builder.create<func::CallOp>(UnknownLoc::get(context),
                                              newFunc1, callOp1Operands);
  // collect return's operands
  llvm::SmallVector<Value> returnOperands;
  for (size_t i = 0; i < mapping.size(); i++) {
    if (mapping[i] < 0)
      returnOperands.push_back(callOp0.getResults()[i]);
  }
  returnOperands.insert(returnOperands.end(), callOp1.getResults().begin(),
                        callOp1.getResults().end());
  builder.create<func::ReturnOp>(UnknownLoc::get(context), returnOperands);

  if (hasEntryPoint) {
    newFunc0->removeAttr(getByteIREntryPointName());
    newFunc1->removeAttr(getByteIREntryPointName());
    NamedAttribute newInputsAttr = NamedAttribute(
        builder.getStringAttr("inputs"),
        builder.getArrayAttr(getMainFuncInputNames(func0, func1, mapping)));
    NamedAttribute newOutputsAttr = NamedAttribute(
        builder.getStringAttr("outputs"),
        builder.getArrayAttr(getMainFuncOutputNames(func0, func1, mapping)));
    mainFunc->setAttr(
        getByteIREntryPointName(),
        DictionaryAttr::get(context, {newInputsAttr, newOutputsAttr}));
  }

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
    llvm::errs() << "module0 must only have one function\n";
    return nullptr;
  }
  if (llvm::count_if(module1.getOps<func::FuncOp>(),
                     [](func::FuncOp func) { return true; }) != 1) {
    llvm::errs() << "module1 must only have one function\n";
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
    func::FuncOp func0 = *module0.getOps<func::FuncOp>().begin();
    func::FuncOp func1 = *module1.getOps<func::FuncOp>().begin();
    // get inputs and outputs name from byteir.entry_point
    auto func0EntryPointDict =
        func0->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
    auto func1EntryPointDict =
        func1->getAttrOfType<DictionaryAttr>(getByteIREntryPointName());
    assert(func0EntryPointDict && func1EntryPointDict &&
           "byteir.entry_point should be dict attr.");
    auto func0InputNamesAttr =
        cast<mlir::ArrayAttr>(func0EntryPointDict.get("inputs"));
    auto func0OutputNamesAttr =
        cast<mlir::ArrayAttr>(func0EntryPointDict.get("outputs"));
    auto func1InputNamesAttr =
        cast<mlir::ArrayAttr>(func1EntryPointDict.get("inputs"));
    auto func1OutputNamesAttr =
        cast<mlir::ArrayAttr>(func1EntryPointDict.get("outputs"));
    SmallVector<std::string> func0OutputNames =
        llvm::to_vector(llvm::map_range(func0OutputNamesAttr, [&](Attribute i) {
          return cast<StringAttr>(i).getValue().str();
        }));
    SmallVector<std::string> func1InputNames =
        llvm::to_vector(llvm::map_range(func1InputNamesAttr, [&](Attribute i) {
          return cast<StringAttr>(i).getValue().str();
        }));
    llvm::SmallVector<int64_t> mapping(func0OutputNames.size(), -1);
    for (size_t i = 0; i < func0OutputNames.size(); i++) {
      // TODO: use unorded_map to speed up
      for (size_t j = 0; j < func1InputNames.size(); j++) {
        if (func0OutputNames[i] == func1InputNames[j]) {
          mapping[i] = j;
        }
      }
    }
    if (all_of(mapping, [](int64_t i) { return i == -1; })) {
      llvm::errs() << "at least one name must be mapped\n";
      return nullptr;
    }
    return mergeTwoModules(module0, module1, context, mapping, true);
  } else if (module0FuncCountWithEntryPoint == 0 &&
             module1FuncCountWithEntryPoint == 0) {
    func::FuncOp func0 = *module0.getOps<func::FuncOp>().begin();
    llvm::SmallVector<int64_t> mapping =
        llvm::to_vector(llvm::seq<int64_t>(0, func0.getNumResults()));
    return mergeTwoModules(module0, module1, context, mapping, false);
  }
  llvm::errs()
      << "func0 and func1 must both have byteir.entry_point or both not have\n";
  return nullptr;
}

OwningOpRef<ModuleOp>
mlir::mergeTwoModulesByMapping(ModuleOp module0, ModuleOp module1,
                               llvm::ArrayRef<int64_t> mapping) {
  assert(mapping.size() != 0 && "mapping size must > 0");
  assert(module0.getContext() == module1.getContext() &&
         "module0 and module1 should have same context");
  MLIRContext *context = module0.getContext();

  // only support module with one function
  if (llvm::count_if(module0.getOps<func::FuncOp>(),
                     [](func::FuncOp func) { return true; }) != 1) {
    llvm::errs() << "module0 must only have one function\n";
    return nullptr;
  }
  if (llvm::count_if(module1.getOps<func::FuncOp>(),
                     [](func::FuncOp func) { return true; }) != 1) {
    llvm::errs() << "module1 must only have one function\n";
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
    return mergeTwoModules(module0, module1, context, mapping, true);
  } else if (module0FuncCountWithEntryPoint == 0 &&
             module1FuncCountWithEntryPoint == 0) {
    return mergeTwoModules(module0, module1, context, mapping, false);
  }
  llvm::errs()
      << "func0 and func1 must both have byteir.entry_point or both not have\n";
  return nullptr;
}
