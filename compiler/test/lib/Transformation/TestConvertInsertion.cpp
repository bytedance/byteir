//===- TestConvertInsertion.cpp ------------------------------------------===//
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

#include "byteir/Dialect/mhlo/Transforms/ConvertInsertion.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace {

constexpr StringRef getByteIRUnitTestAttrName() {
  return "__byteir_unit_test__";
}

std::unique_ptr<ConvertRuleBase> unitTestRuleExample(MLIRContext *ctx) {
  auto collector = std::make_unique<ConvertOnlyCheckElementType>(
      getByteIRUnitTestAttrName());
  collector->convertElementType.try_emplace(Float32Type::get(ctx),
                                            Float16Type::get(ctx));
  return collector;
}

struct TestConvertInsertionPass
    : public PassWrapper<TestConvertInsertionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertInsertionPass)

  StringRef getArgument() const final { return "test-insert-convert"; }

  StringRef getDescription() const final {
    return "Test Insert ConvertOps (for CallOps and FuncOps now)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto m = getOperation();
    auto ctx = m.getContext();
    std::unique_ptr<ConvertRuleBase> testRule = unitTestRuleExample(ctx);

    OpPassManager pm(m.getOperationName());
    pm.addPass(createConvertInsertionPass(testRule.get()));
    addCleanUpPassPipeline(pm);
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

struct TestI16ConvertRule : public ConvertRuleBase {

  explicit TestI16ConvertRule(mlir::StringRef strRef)
      : anchorAttr(strRef.str()) {}
  virtual ~TestI16ConvertRule() {}
  bool checkFunc(func::FuncOp func) { return func->hasAttr(anchorAttr); }

  std::optional<mlir::TensorType> checkArg(func::FuncOp func, size_t offset,
                                           bool isArg) {
    auto context = func.getContext();
    auto builder = std::make_unique<mlir::OpBuilder>(context);
    FunctionType funcType = func.getFunctionType();
    if (isArg) {
      auto TensorTy = funcType.getInput(offset).dyn_cast<TensorType>();
      mlir::Type I16Type;
      if (offset == 0) {
        I16Type = builder.get()->getIntegerType(16);
        return TensorTy.clone(I16Type);
      } else
        return TensorTy;
    } else {
      auto TensorTy = funcType.getResult(offset).dyn_cast<TensorType>();
      mlir::Type I16Type;
      if (offset == 0) {
        I16Type = builder.get()->getIntegerType(16);
        return TensorTy.clone(I16Type);
      } else
        return TensorTy;
    }
  }

  std::string anchorAttr;
};

struct TestCustomConvertPass
    : public PassWrapper<TestCustomConvertPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCustomConvertPass)

  StringRef getArgument() const final { return "test-custom-convert"; }

  StringRef getDescription() const final {
    return "Test custom convert rules to insert ConvertOps (for CallOps and "
           "FuncOps now)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto m = getOperation();
    auto testRule =
        std::make_unique<TestI16ConvertRule>(getByteIRUnitTestAttrName());

    OpPassManager pm(m.getOperationName());
    pm.addPass(createConvertInsertionPass(testRule.get()));
    addCleanUpPassPipeline(pm);
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestConvertInsertionPass() {
  PassRegistration<TestConvertInsertionPass>();
}
void registerTestCustomConvertPass() {
  PassRegistration<TestCustomConvertPass>();
}
} // namespace test
} // namespace byteir
