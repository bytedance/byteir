//===- StaticShapeInference.cpp -------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/StaticShapeInference.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct StaticShapeInferencePass
    : public StaticShapeInferenceBase<StaticShapeInferencePass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    (void)runShapeInference(funcOp, /*isStaticShapeInfer=*/false);
  };
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createStaticShapeInferencePass() {
  return std::make_unique<StaticShapeInferencePass>();
}
