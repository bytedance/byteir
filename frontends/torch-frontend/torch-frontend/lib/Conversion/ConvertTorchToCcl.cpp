//===- ConvertTorchToCcl.cpp ----------------------------------*--- C++ -*-===//
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

#include "torch-frontend/Conversion/ConvertTorchToCcl.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertTorchToCcl : public ConvertTorchToCclBase<ConvertTorchToCcl> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
    registry.insert<ccl::CclDialect>();
  }

  void runOnOperation() override {}
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createConvertTorchToCcl() {
  return std::make_unique<ConvertTorchToCcl>();
}
