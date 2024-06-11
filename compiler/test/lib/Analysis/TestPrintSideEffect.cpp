//===- TestPrintSideEffect.cpp --------------------------------------------===//
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

#include "byteir/Analysis/SideEffect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace byteir;

namespace {

struct TestPrintArgSideEffectPass
    : public PassWrapper<TestPrintArgSideEffectPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintArgSideEffectPass)

  StringRef getArgument() const final { return "test-print-arg-side-effect"; }

  StringRef getDescription() const final { return "Print the arg side effect"; }

  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override {
    auto &os = llvm::outs();
    ModuleOp m = getOperation();
    ArgSideEffectAnalysis analysis;
    analysis.dump(llvm::outs());
    os << "============= Test Module"
       << " =============\n";
    for (auto f : m.getOps<func::FuncOp>()) {
      for (auto &block : f.getBlocks()) {
        for (auto &op : block.without_terminator()) {
          os << "Testing " << op.getName() << ":\n";
          for (unsigned i = 0; i < op.getNumOperands(); ++i) {
            auto argSETy = analysis.getType(&op, i);
            os << "arg " << i << " ArgSideEffectType: " << str(argSETy) << "\n";
          }
        }
      }
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestPrintArgSideEffectPass() {
  PassRegistration<TestPrintArgSideEffectPass>();
}
} // namespace test
} // namespace byteir
