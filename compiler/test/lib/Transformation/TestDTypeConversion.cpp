//===- TestDTypeConversion.cpp ------------------------------------------===//
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

#include "byteir/Dialect/mhlo/Transforms/DTypeConversion.h"
#include "byteir/Utils/PipelineUtils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {

constexpr StringRef getByteIRUnitTestAttrName() {
  return "__byteir_unit_test__";
}

struct TestF32toF16ConvertRule : public DTypeConvertRuleBase {

  explicit TestF32toF16ConvertRule(mlir::StringRef strRef,
                                   mlir::MLIRContext *ctx)
      : anchorAttr(strRef.str()) {
    std::vector<Type> f16BinaryInput = {Float16Type::get(ctx),
                                        Float16Type::get(ctx)};
    std::vector<Type> f16BinaryOutput = {Float16Type::get(ctx)};
    std::pair<std::vector<Type>, std::vector<Type>> f16Binary =
        std::make_pair(f16BinaryInput, f16BinaryOutput);

    std::vector<Type> mixPBinaryInput = {Float32Type::get(ctx),
                                         Float16Type::get(ctx)};
    std::vector<Type> mixPBinaryOutput = {Float16Type::get(ctx)};
    std::pair<std::vector<Type>, std::vector<Type>> mixPBinary =
        std::make_pair(mixPBinaryInput, mixPBinaryOutput);

    convertRules["mhlo.add"].push_back(f16Binary);
    convertRules["mhlo.reduce_window"].push_back(f16Binary);
    convertRules["mhlo.maximum"].push_back(f16Binary);
    // custom call target name
    convertRules["f16_custom_call"].push_back(f16Binary);
    convertRules["mixp_custom_call"].push_back(mixPBinary);
  }
  virtual ~TestF32toF16ConvertRule() {}
  bool checkFunc(func::FuncOp func) override {
    return func->hasAttr(anchorAttr);
  }

  std::string anchorAttr;
};

struct TestModifyFuncConvertRule : public TestF32toF16ConvertRule {
  using TestF32toF16ConvertRule::TestF32toF16ConvertRule;
  bool canModifyFuncArg(func::FuncOp) override { return true; }
};

struct TestDTypeConversionPass
    : public PassWrapper<TestDTypeConversionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDTypeConversionPass)

  StringRef getArgument() const final { return "test-dtype-convert"; }

  StringRef getDescription() const final {
    return "Test custom data type convert rules";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto m = getOperation();
    auto testRule = std::make_unique<TestF32toF16ConvertRule>(
        getByteIRUnitTestAttrName(), m->getContext());

    OpPassManager pm(m.getOperationName());
    pm.addPass(createDTypeConversionPass(testRule.get()));
    addCleanUpPassPipeline(pm);
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

struct TestDTypeConversionModifyFuncPass
    : public PassWrapper<TestDTypeConversionModifyFuncPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestDTypeConversionModifyFuncPass)

  StringRef getArgument() const final {
    return "test-dtype-convert-modify-func";
  }

  StringRef getDescription() const final {
    return "Test custom data type convert rules that can change func signature";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    auto m = getOperation();
    auto testRule = std::make_unique<TestModifyFuncConvertRule>(
        getByteIRUnitTestAttrName(), m->getContext());

    OpPassManager pm(m.getOperationName());
    pm.addPass(createDTypeConversionPass(testRule.get()));
    addCleanUpPassPipeline(pm);
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestDTypeConversionPass() {
  PassRegistration<TestDTypeConversionPass>();
  PassRegistration<TestDTypeConversionModifyFuncPass>();
}
} // namespace test
} // namespace byteir
