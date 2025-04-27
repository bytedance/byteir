//===- InsertInputShapeConstraint.cpp -------------------------------C++ --===//
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

#include "byteir/Dialect/Shape/Transforms/InsertInputShapeConstraint.h"

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <string>

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct InsertInputShapeConstraintPass
    : public InsertInputShapeConstraintBase<InsertInputShapeConstraintPass> {
  InsertInputShapeConstraintPass(llvm::StringRef mode) {
    this->mode = mode.str();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    if (mode == "all-dynamic-batch-same") {
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      Value firstDynamicBatch = nullptr;
      for (auto arg : funcOp.getArguments()) {
        auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
        if (!tensorType)
          continue;
        if (tensorType.hasStaticShape())
          continue;
        if (tensorType.getRank() <= 0)
          continue;
        if (!tensorType.isDynamicDim(0))
          continue;
        auto loc = builder.getUnknownLoc();
        Value dim = builder.create<tensor::DimOp>(loc, arg, 0);
        if (firstDynamicBatch == nullptr) {
          firstDynamicBatch = dim;
        } else {
          builder.create<shape_ext::MeetOp>(loc, dim, firstDynamicBatch);
        }
      }
    } else {
      funcOp->emitOpError("unknown mode: ") << mode;
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createInsertInputShapeConstraintPass(llvm::StringRef mode) {
  return std::make_unique<InsertInputShapeConstraintPass>(mode);
}
