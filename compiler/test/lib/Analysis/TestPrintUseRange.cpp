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
#include "byteir/Analysis/UseRange.h"
#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace byteir;

namespace {

struct TestPrintUseRangePass
    : public PassWrapper<TestPrintUseRangePass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintUseRangePass)

  StringRef getArgument() const final { return "test-print-use-range"; }

  StringRef getDescription() const final {
    return "Print the contents of a constructed use range information.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::lmhlo::LmhloDialect>();
  }
  void runOnOperation() override {
    llvm::outs() << "Testing : " << getOperation().getName() << "\n";
    auto op = getOperation();
    byteir::Liveness liveness(op);
    UserangeAnalysis(op, &liveness,
                     bufferization::BufferPlacementAllocs(getOperation()),
                     BufferViewFlowAnalysis(getOperation()))
        .dump(llvm::outs());
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestPrintUseRangePass() {
  PassRegistration<TestPrintUseRangePass>();
}
} // namespace test
} // namespace byteir