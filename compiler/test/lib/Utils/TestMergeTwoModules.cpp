//===- TestMergeTwoModules.cpp --------------------------------------------===//
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

#include "byteir/Utils/ModuleUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

#include <string>

using namespace mlir;

namespace {

struct TestMergeTwoModulesPass
    : public PassWrapper<TestMergeTwoModulesPass,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMergeTwoModulesPass)

  TestMergeTwoModulesPass()
      : PassWrapper<TestMergeTwoModulesPass, OperationPass<mlir::ModuleOp>>() {}

  TestMergeTwoModulesPass(const TestMergeTwoModulesPass &other)
      : PassWrapper<TestMergeTwoModulesPass, OperationPass<mlir::ModuleOp>>(
            other) {}

  StringRef getArgument() const final { return "test-merge-two-modules"; }

  StringRef getDescription() const final {
    return "Test createTestMergeTwoModulesImpl()";
  }

  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override {
    auto module0Op = getOperation();
    MLIRContext *context = &getContext();

    std::string errorMessage;
    auto secondModuleInput =
        mlir::openInputFile(this->secondModulePath, &errorMessage);
    if (!secondModuleInput) {
      llvm::errs() << "can't open file: " << this->secondModulePath << "\n";
      return signalPassFailure();
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(secondModuleInput), llvm::SMLoc());
    auto module1Op = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context);

    mlir::OwningOpRef<mlir::ModuleOp> m =
        mergeTwoModulesByNameOrOrder(module0Op, module1Op.get());
    if (!m) {
      llvm::errs() << "can't merge these two modules\n";
      return signalPassFailure();
    }
    IRMapping emptyMap;
    module0Op.getBodyRegion().takeBody(m.get().getBodyRegion());
  }

protected:
  mlir::Pass::Option<std::string> secondModulePath{
      *this, "second-module-path", llvm::cl::desc("Second Module Path.")};
};

} // namespace

namespace byteir {
namespace test {
void registerTestMergeTwoModulesPass() {
  PassRegistration<TestMergeTwoModulesPass>();
}
} // namespace test
} // namespace byteir
