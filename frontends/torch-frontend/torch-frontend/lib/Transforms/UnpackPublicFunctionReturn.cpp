//===- UnpackPublicFunctionReturn.cpp ---------------------------*- C++ -*-===//
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

#include "torch-frontend/Transforms/UnpackPublicFunctionReturn.h"
#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// note: only convert return list to return tuple now.
struct UnpackPublicFunctionReturnPass
    : public UnpackPublicFunctionReturnBase<UnpackPublicFunctionReturnPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();

    if (!funcOp.isPublic()) {
      return;
    }
    // TODO: check func has no callsite

    if (funcOp.getResultTypes().size() != 1) {
      return;
    }
    bool hasListType = llvm::any_of(funcOp.getResultTypes(), [](Type ty) {
      return isa<Torch::ListType>(ty);
    });
    if (!hasListType) {
      return;
    }

    func::ReturnOp returnOp =
        llvm::cast<func::ReturnOp>(&funcOp.front().back());
    SmallVector<Value> newResults;
    SmallVector<Type> newResultTypes;
    for (auto operand : returnOp.getOperands()) {
      if (isa<Torch::ListType>(operand.getType())) {
        auto primListConstructOp =
            operand.getDefiningOp<Torch::PrimListConstructOp>();
        if (primListConstructOp) {
          for (auto listOperand : primListConstructOp.getOperands()) {
            newResults.push_back(listOperand);
            newResultTypes.push_back(listOperand.getType());
          }
          continue;
        }
      }
      newResults.push_back(operand);
      newResultTypes.push_back(operand.getType());
    }
    OpBuilder rewriter(returnOp);
    auto loc = returnOp->getLoc();
    auto tupleType = Torch::TupleType::get(ctx, newResultTypes);
    auto tupleOp = rewriter.create<Torch::PrimTupleConstructOp>(loc, tupleType,
                                                                newResults);
    rewriter.create<func::ReturnOp>(loc, tupleOp.getResult());
    returnOp->erase();
    funcOp.setType(
        FunctionType::get(ctx, funcOp.getArgumentTypes(), tupleType));
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createUnpackPublicFunctionReturnPass() {
  return std::make_unique<UnpackPublicFunctionReturnPass>();
}
