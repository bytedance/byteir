//===- ConvertFuncToCustomCall.h ------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
class NamedAttrList;
class Value;
class ValueRange;
class TypeRange;

// the abstract class of FuncToCustomCallConverter
// Some member functions are implemented for trival cases
struct FuncToCustomCallConverterBase {
  FuncToCustomCallConverterBase(){};
  virtual ~FuncToCustomCallConverterBase() {}

  virtual bool checkFunc(func::FuncOp) = 0;

  virtual NamedAttrList getAttrs(func::FuncOp) = 0;

  virtual TypeRange getResultTypes(func::FuncOp func) {
    return func.getResultTypes();
  }

  virtual ValueRange getOperands(func::CallOp call) {
    return call.getOperands();
  }

  virtual unsigned getNewResultIdx(func::CallOp, unsigned oldIdx) {
    return oldIdx;
  }

  virtual std::function<void(func::FuncOp, ModuleOp)>
      getCustomizedConversion(func::FuncOp) = 0;
};

// a common CustomMeta for creating CustomCall
struct CustomLoopupMeta {
  std::string callTargetName;
  bool hasSideEffect;
  bool useDefault;
  SmallVector<unsigned> opernadOldIndices; // new id to old id
  SmallVector<unsigned> resultOldIndices;  // new id to old id
  SmallVector<unsigned> resultNewIndices;  // old id to new id

  CustomLoopupMeta()
      : callTargetName(""), hasSideEffect(false), useDefault(true) {}

  CustomLoopupMeta(const std::string &tagetName, bool sideEffect)
      : callTargetName(tagetName), hasSideEffect(sideEffect), useDefault(true) {
  }

  CustomLoopupMeta(const std::string &tagetName, bool sideEffect,
                   ArrayRef<unsigned> operand, ArrayRef<unsigned> resultOld,
                   ArrayRef<unsigned> resultNew)
      : callTargetName(tagetName), hasSideEffect(sideEffect), useDefault(false),
        opernadOldIndices(operand.begin(), operand.end()),
        resultOldIndices(resultOld.begin(), resultOld.end()),
        resultNewIndices(resultNew.begin(), resultNew.end()) {}
};

// a common FuncToCustomCallConverter using Lookup
struct FuncToCustomCallConverterLookup : public FuncToCustomCallConverterBase {

  FuncToCustomCallConverterLookup() : FuncToCustomCallConverterBase() {}

  explicit FuncToCustomCallConverterLookup(
      const llvm::StringMap<CustomLoopupMeta> &externalMap)
      : FuncToCustomCallConverterBase(), funcNameToCustomMeta(externalMap) {}

  virtual ~FuncToCustomCallConverterLookup() {}

  bool checkFunc(func::FuncOp) override;

  NamedAttrList getAttrs(func::FuncOp) override;

  TypeRange getResultTypes(func::FuncOp) override;

  ValueRange getOperands(func::CallOp) override;

  unsigned getNewResultIdx(func::CallOp, unsigned) override;

  std::function<void(func::FuncOp, ModuleOp)>
      getCustomizedConversion(func::FuncOp) override;

  llvm::StringMap<CustomLoopupMeta> funcNameToCustomMeta;

  llvm::StringMap<std::function<void(func::FuncOp, ModuleOp)>>
      funcNameToCustomizedConversion;
};

std::unique_ptr<OperationPass<ModuleOp>>
createConvertFuncToCustomCallPass(FuncToCustomCallConverterBase *converter);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTFUNCTOCUSTOMCALL_H