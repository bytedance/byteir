//===- TestByreOpInterface.cpp --------------------------------------------===//
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

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

struct TestByreOpInterfacePass
    : public PassWrapper<TestByreOpInterfacePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestByreOpInterfacePass)

  StringRef getArgument() const final { return "test-byre-op-interface"; }

  StringRef getDescription() const final { return "Test byre op interface"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<byre::ByreDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    op.walk([&](byre::ByreOp op) {
      llvm::outs() << op.getCalleeName() << '\n';
      auto inputs = op.getInputs();
      llvm::outs() << inputs.size() << ' ' << "Inputs:\n";
      for (auto &&input : inputs) {
        llvm::outs() << '\t' << input << '\n';
      }
      auto outputs = op.getOutputs();
      llvm::outs() << outputs.size() << ' ' << "Outputs:\n";
      for (auto &&output : outputs) {
        llvm::outs() << '\t' << output << '\n';
      }
    });
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestByreOpInterfacePass() {
  PassRegistration<TestByreOpInterfacePass>();
}
} // namespace test
} // namespace byteir
