//===- TestPrintSymbolicShape.cpp -----------------------------------------===//
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

#include "byteir/Analysis/SymbolicShape.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestPrintSymbolicShapePass
    : public PassWrapper<TestPrintSymbolicShapePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintSymbolicShapePass)

  StringRef getArgument() const final { return "test-print-symbolic-shape"; }

  StringRef getDescription() const final {
    return "Print the symbolic shape auxiliary functions.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::shape::ShapeDialect>();
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    SymbolicShapeAnalysis(op).dump(llvm::outs());
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestPrintSymbolicShapePass() {
  PassRegistration<TestPrintSymbolicShapePass>();
}
} // namespace test
} // namespace byteir