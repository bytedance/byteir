//===- OFCanonicalizer.cpp ------------------------------------------------===//
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

#include "onnx-frontend/src/Conversion/OFCanonicalizer.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct OFCanonicalizerPass
    : public onnx_frontend::OFCanonicalizerBase<OFCanonicalizerPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OFCanonicalizerPass)

  OFCanonicalizerPass() {}

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    SmallVector<std::string> disabledPatterns{
        "FuseBatchNormInferenceModeConvPattern",
        "RewriteBatchNormInferenceModeConvPattern1",
        "RewriteBatchNormInferenceModeConvPattern2"};
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    patterns =
        FrozenRewritePatternSet(std::move(owningPatterns), disabledPatterns,
                                /*enabledPatterns*/ {});
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Canonicalization
    // TODO: revert this after bump llvm version after 2024-12-22
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    LogicalResult converged =
        applyPatternsAndFoldGreedily(module, patterns, config);
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (testConvergence && failed(converged))
      signalPassFailure();

    // Correct the function's output types
    module.walk([&](func::FuncOp f) {
      auto &funcBody = f.getBody();
      // Check if a terminator op exists for function.
      if (!funcBody.empty() && !funcBody.back().empty() &&
          funcBody.back().back().hasTrait<OpTrait::IsTerminator>())
        if (auto returnOp = f.getBody().back().getTerminator()) {
          auto results = returnOp->getOperandTypes();
          f.setType(FunctionType::get(
              f.getContext(), f.getFunctionType().getInputs(),
              std::vector<Type>(results.begin(), results.end())));
        }
    });
  }

  mlir::FrozenRewritePatternSet patterns;
};
} // namespace

namespace onnx_frontend {
std::unique_ptr<mlir::Pass> createOFCanonicalizerPass() {
  return std::make_unique<OFCanonicalizerPass>();
}
} // namespace onnx_frontend
