//===- AnchoredPipeline.cpp -----------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/AnchoredPipeline.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct AnchoredPipelinePass
    : public AnchoredPipelineBase<AnchoredPipelinePass> {

  explicit AnchoredPipelinePass(const std::string &anchor)
      : AnchoredPipelineBase<AnchoredPipelinePass>(), pm("op") {
    this->anchorAttr = anchor;
  }

  AnchoredPipelinePass(const std::string &anchor, OpPassManager &otherPM)
      : AnchoredPipelineBase<AnchoredPipelinePass>(), pm(otherPM) {
    this->anchorAttr = anchor;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    pm.getDependentDialects(registry);
  }

  void runOnOperation() override {
    if (anchorAttr.empty()) {
      return;
    }

    auto op = getOperation();

    if (!op->hasAttr(anchorAttr)) {
      return;
    }

    if (mlir::failed(runPipeline(pm, op))) {
      signalPassFailure();
    }
  }

  OpPassManager pm;
};

} // namespace

std::unique_ptr<Pass>
mlir::createAnchoredPipelinePass(llvm::StringRef anchorTag,
                                 OpPassManager &otherPM) {
  return std::make_unique<AnchoredPipelinePass>(anchorTag.str(), otherPM);
}

std::unique_ptr<Pass>
mlir::createAnchoredPipelinePass(llvm::StringRef anchorTag) {
  return std::make_unique<AnchoredPipelinePass>(anchorTag.str());
}
