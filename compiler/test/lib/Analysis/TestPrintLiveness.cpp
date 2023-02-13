/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "byteir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace byteir;

namespace {

struct TestPrintLivenessPass
    : public PassWrapper<TestPrintLivenessPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintLivenessPass)

  StringRef getArgument() const final { return "test-print-liveness"; }
  StringRef getDescription() const final {
    return "Print the contents of a constructed liveness information.";
  }
  void runOnOperation() override {
    llvm::outs() << "Testing : " << getOperation().getName() << "\n";
    getAnalysis<Liveness>().print(llvm::outs());
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestPrintLivenessPass() {
  PassRegistration<TestPrintLivenessPass>();
}
} // namespace test
} // namespace byteir