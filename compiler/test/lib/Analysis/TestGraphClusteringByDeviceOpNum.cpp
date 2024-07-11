//===- TestGraphClusteringByDeviceOpNum.cpp -------------------------------===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "byteir/Transforms/GraphClusteringByDevice.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace {

struct TestGraphClusteringByDeviceOpNum
    : public PassWrapper<TestGraphClusteringByDeviceOpNum,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGraphClusteringByDeviceOpNum)

  TestGraphClusteringByDeviceOpNum() = default;

  TestGraphClusteringByDeviceOpNum(const TestGraphClusteringByDeviceOpNum &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final {
    return "test-graph-clustering-by-device-op-num";
  }

  StringRef getDescription() const final {
    return "Clustering device sub-graph and validate by op num";
  }

  void getDependentDialects(DialectRegistry &registry) const override {}

  Option<int64_t> opNum{*this, "op-num",
                        llvm::cl::desc("the minimum size of sub-graph"),
                        llvm::cl::init(1)};

  void runOnOperation() override {
    ModuleOp m = getOperation();
    auto validateSubGraphFn = [&](llvm::ArrayRef<Operation *> ops) -> bool {
      return ops.size() >= opNum;
    };

    if (failed(GraphClustingByDevice(
            m, "device", "test", "__byteir_test_device__", false, false,
            GraphClusteringAlgo::kGreedy,
            /*enableMultiGraph=*/true, validateSubGraphFn))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestGraphClusteringByDeviceOpNumPass() {
  PassRegistration<TestGraphClusteringByDeviceOpNum>();
}
} // namespace test
} // namespace byteir
