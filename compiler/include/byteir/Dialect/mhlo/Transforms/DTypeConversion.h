//===- DtypeConversion.h --------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_DTYPECONVERSION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_DTYPECONVERSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class Operation;
class Value;
class TensorType;
class ModuleOp;

// abstract struct for convert rule
struct DTypeConvertRuleBase {

  DTypeConvertRuleBase(){};
  virtual ~DTypeConvertRuleBase() {}

  // default all function
  virtual bool checkFunc(func::FuncOp) { return true; }

  // default all function
  virtual bool canModifyFuncArg(func::FuncOp) { return false; }

  // Data type rules for operations
  llvm::DenseMap<llvm::StringRef,
                 std::vector<std::pair<std::vector<Type>, std::vector<Type>>>>
      convertRules;
};

// use DTypeConvertRuleBase to decide how to convert data types
std::unique_ptr<OperationPass<ModuleOp>>
createDTypeConversionPass(DTypeConvertRuleBase *);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_DTYPECONVERSION_H
