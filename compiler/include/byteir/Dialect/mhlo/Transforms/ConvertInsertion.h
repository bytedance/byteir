//===- ConvertInsertion.h -------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTINSERTION_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTINSERTION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class Operation;
class Value;
class TensorType;
class ModuleOp;

// abstract struct for convert rule
struct ConvertRuleBase {

  ConvertRuleBase(){};
  virtual ~ConvertRuleBase() {}

  // default all function
  virtual bool checkFunc(func::FuncOp) { return true; }

  virtual std::optional<mlir::TensorType>
  checkArg(func::FuncOp func, size_t offset, bool isArg) = 0;
};

// a common ConvertRule
// using only ElementType in registered convertElementType for checkArg
// and only anchorAttr for checkFunc
struct ConvertOnlyCheckElementType : public ConvertRuleBase {

  explicit ConvertOnlyCheckElementType(mlir::StringRef);

  virtual ~ConvertOnlyCheckElementType() {}

  virtual bool checkFunc(func::FuncOp) override;
  std::optional<mlir::TensorType> checkArg(func::FuncOp func, size_t offset,
                                           bool isArg) override;

  // convert element type from first to second
  llvm::DenseMap<mlir::Type, mlir::Type> convertElementType;

  std::string anchorAttr;
};

// use ConvertRuleBase to decide how to insert convert
std::unique_ptr<OperationPass<ModuleOp>>
createConvertInsertionPass(ConvertRuleBase *);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CONVERTINSERTION_H
