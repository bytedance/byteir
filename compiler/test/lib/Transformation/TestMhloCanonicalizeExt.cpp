//===- TestMhloCanonicalizeExt.cpp ----------------------------------------===//
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

#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

struct TestMhloCanonicalizeExtPass
    : public PassWrapper<TestMhloCanonicalizeExtPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMhloCanonicalizeExtPass)

  StringRef getArgument() const final { return "test-mhlo-canonicalize-ext"; }

  StringRef getDescription() const final { return "Mhlo Canonicalize Ext"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
    registry.insert<mlir::shape::ShapeDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    mhlo::populateCanonicalizeExtPatterns(patterns, context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestMhloCanonicalizeExtPass() {
  PassRegistration<TestMhloCanonicalizeExtPass>();
}
} // namespace test
} // namespace byteir
