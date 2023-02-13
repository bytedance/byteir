//===- FuncArgRearrangement.h ---------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_FUNCARGREARRANGEMENT_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_FUNCARGREARRANGEMENT_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <string>

namespace mlir {
class FunctionType;
class ModuleOp;
class OpBuilder;
class Value;

namespace func {
class FuncOp;
} // namespace func

// abstract base class of FuncArgRearranger
class FuncArgRearrangerBase {
public:
  FuncArgRearrangerBase() {}

  virtual ~FuncArgRearrangerBase() {}

  virtual bool init() = 0;

  // get or create a new func
  virtual func::FuncOp getOrCreateNewFunc(OpBuilder &b) = 0;

  // get or create a FuncArg
  virtual Value getOrCreateNewFromOldFuncArg(OpBuilder &b, unsigned newId,
                                             ArrayRef<Value> oldValues) = 0;

  // note old arg might have many or non ways to be constructed from new ones
  virtual llvm::SmallVector<Value>
  getOrCreateOldFromNewFuncArg(OpBuilder &b, unsigned oldId,
                               ArrayRef<Value> newValues) = 0;

  // get or create a FuncResult
  virtual Value getOrCreateNewFromOldFuncResult(OpBuilder &b, unsigned newId,
                                                ArrayRef<Value> oldValues) = 0;

  // note old result might have many or non ways to be constructed from new ones
  virtual llvm::SmallVector<Value>
  getOrCreateOldFromNewFuncResult(OpBuilder &b, unsigned oldId,
                                  ArrayRef<Value> newValues) = 0;
};

// abstract base class of FuncArgRearrangerBuilder
class FuncArgRearrangerBuilderBase {
public:
  FuncArgRearrangerBuilderBase() {}
  virtual ~FuncArgRearrangerBuilderBase() {}

  virtual std::unique_ptr<FuncArgRearrangerBase>
  createFuncArgRearranger(func::FuncOp f) = 0;
};

std::unique_ptr<OperationPass<ModuleOp>>
createFuncArgRearrangementPass(FuncArgRearrangerBuilderBase *builder,
                               const std::string &anchor = "",
                               bool keepAnchor = false);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_FUNCARGREARRANGEMENT_H
