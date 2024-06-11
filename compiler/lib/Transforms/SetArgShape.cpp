//===- SetArgShape.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/SetArgShape.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

#include "./PassDetail.h"

#define DEBUG_TYPE "set-arg-shape-pass"

using namespace mlir;

namespace {

struct SetArgShapePass : public SetArgShapeBase<SetArgShapePass> {

  SetArgShapePass() = default;

  SetArgShapePass(int dim, int size, std::string entryFuncName,
                  std::string argAttrName)
      : SetArgShapeBase<SetArgShapePass>::SetArgShapeBase() {
    this->dim = dim;
    this->size = size;
    this->entryFuncName = entryFuncName;
    this->argAttrName = argAttrName;
  }

  SetArgShapePass(int dim, int size, std::string entryFuncName,
                  std::function<bool(BlockArgument)> shouldSetShape)
      : SetArgShapeBase<SetArgShapePass>::SetArgShapeBase() {
    this->dim = dim;
    this->size = size;
    this->entryFuncName = entryFuncName;
    this->shouldSetShape = shouldSetShape;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp funcOp =
        module.lookupSymbol<func::FuncOp>(this->entryFuncName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot find the speficied function. This "
                                 "pass will be ignored.\n");
      return;
    }
    if (this->dim < 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Dim is less than zero. This pass will be ignored.\n");
      return;
    }

    FunctionType funcType = funcOp.getFunctionType();
    SmallVector<Type, 4> newArgTypes;
    newArgTypes.reserve(funcOp.getNumArguments());
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      auto arg = funcOp.getArgument(i);
      if ((this->shouldSetShape && this->shouldSetShape(arg)) ||
          funcOp.getArgAttr(i, this->argAttrName)) {
        if (auto inputTy = dyn_cast<RankedTensorType>(arg.getType())) {
          Type elementType = inputTy.getElementType();
          llvm::SmallVector<int64_t> shape(inputTy.getShape().begin(),
                                           inputTy.getShape().end());
          if (this->dim < int(shape.size())) {
            shape[this->dim] = this->size;
            auto newArgType = RankedTensorType::get(shape, elementType,
                                                    inputTy.getEncoding());
            if (newArgType != inputTy) {
              arg.setType(newArgType);
            }
          }
        } else if (auto inputTy = dyn_cast<MemRefType>(arg.getType())) {
          Type elementType = inputTy.getElementType();
          llvm::SmallVector<int64_t> shape(inputTy.getShape().begin(),
                                           inputTy.getShape().end());
          if (this->dim < int(shape.size())) {
            shape[this->dim] = this->size;
            auto newArgType =
                MemRefType::get(shape, elementType, inputTy.getLayout(),
                                inputTy.getMemorySpace());
            if (newArgType != inputTy) {
              arg.setType(newArgType);
            }
          }
        }
      }
      newArgTypes.push_back(arg.getType());
    }
    funcOp.setType(FunctionType::get(funcOp.getContext(), newArgTypes,
                                     funcType.getResults()));
  }

  std::function<bool(BlockArgument)> shouldSetShape;
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createSetArgShapePass() {
  return std::make_unique<SetArgShapePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetArgShapePass(int dim, int size, std::string entryFuncName,
                            std::string argAttrName) {
  return std::make_unique<SetArgShapePass>(dim, size, entryFuncName,
                                           argAttrName);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createSetArgShapePass(int dim, int size, std::string entryFuncName,
                            std::function<bool(BlockArgument)> shouldSetShape) {
  return std::make_unique<SetArgShapePass>(dim, size, entryFuncName,
                                           shouldSetShape);
}
