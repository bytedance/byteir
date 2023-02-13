//===- InsertShapeConstraint.cpp ------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/InsertShapeConstraint.h"
#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include <vector>

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct InsertShapeConstraintPass
    : public InsertShapeConstraintBase<InsertShapeConstraintPass> {
  InsertShapeConstraintPass()
      : InsertShapeConstraintBase<
            InsertShapeConstraintPass>::InsertShapeConstraintBase() {
    registerAllMhloShapeConstraints();
  }

  void runOnOperation() override {
    std::vector<Operation *> ops;
    func::FuncOp funcOp = getOperation();
    funcOp.walk([&](Operation *op) { ops.push_back(op); });

    OpBuilder builder(funcOp->getContext());

    // std::reverse(ops.begin(), ops.end());
    for (Operation *op : ops) {
      llvm::StringRef opName;

      if (auto customCall = llvm::dyn_cast<mhlo::CustomCallOp>(op)) {
        opName = customCall.getCallTargetName();
      } else {
        opName = op->getName().getStringRef();
      }

      if (auto insertShapeConstraintFunc = insertShapeConstraint(opName)) {
        LogicalResult status = insertShapeConstraintFunc(op, builder);
        (void)status; // Suppress unused warning
      }
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createInsertShapeConstraintPass() {
  return std::make_unique<InsertShapeConstraintPass>();
}
