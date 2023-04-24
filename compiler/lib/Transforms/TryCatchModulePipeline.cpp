//===- TryCatchModulePipeline.cpp -----------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/TryCatchModulePipeline.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct TryCatchModulePipelinePass
    : public TryCatchModulePipelineBase<TryCatchModulePipelinePass> {
  explicit TryCatchModulePipelinePass(const std::string &anchor,
                                      TryCatchConfig *config)
      : TryCatchModulePipelineBase<TryCatchModulePipelinePass>(),
        pm(ModuleOp::getOperationName()) {
    this->anchorAttr = anchor;
    if (config) {
      this->stepFunc = config->stepFunc;
      this->copyFunc = config->copyFunc;
      this->maxTries = config->maxTries;
    }
  }
  TryCatchModulePipelinePass(const std::string &anchor, TryCatchConfig *config,
                             OpPassManager &otherPM)
      : TryCatchModulePipelineBase<TryCatchModulePipelinePass>(), pm(otherPM) {
    this->anchorAttr = anchor;
    if (config) {
      this->stepFunc = config->stepFunc;
      this->copyFunc = config->copyFunc;
      this->maxTries = config->maxTries;
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    pm.getDependentDialects(registry);
  }

  void runOnOperation() override {
    if (anchorAttr.empty()) {
      return;
    }

    auto m = getOperation();

    if (!m->hasAttr(anchorAttr)) {
      return;
    }

    // copy the original module{func} op for loop back.
    copyFunc(m);

    int try_times = 0;
    while (try_times++ < maxTries) {
      if (mlir::failed(runPipeline(pm, m))) {
        stepFunc(m);
        copyFunc(m);
      } else {
        return;
      }
    }
    // after retry limit, signal fail.
    signalPassFailure();
  }

  OpPassManager pm;
  std::function<void(ModuleOp &m)> stepFunc = nullptr;
  std::function<void(ModuleOp &m)> copyFunc = nullptr;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createTryCatchModulePipelinePass(
    llvm::StringRef anchorTag, TryCatchConfig *config, OpPassManager &otherPM) {
  return std::make_unique<TryCatchModulePipelinePass>(anchorTag.str(), config,
                                                      otherPM);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createTryCatchModulePipelinePass(llvm::StringRef anchorTag,
                                       TryCatchConfig *config) {
  return std::make_unique<TryCatchModulePipelinePass>(anchorTag.str(), config);
}
