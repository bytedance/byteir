//===- ApplyPDLPatterns.cpp -----------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/ApplyPDLPatterns.h"
#include "byteir/Utils/PatternMatch.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
#define GEN_PASS_DECL_APPLYPDLPATTERNS
#define GEN_PASS_DEF_APPLYPDLPATTERNS
#include "byteir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;

namespace {
struct ApplyPDLPatternsPass
    : public impl::ApplyPDLPatternsBase<ApplyPDLPatternsPass> {
  ApplyPDLPatternsPass(const std::string &pdlFile) : ApplyPDLPatternsBase() {
    this->pdlFile = pdlFile;
  }

  void runOnOperation() override {
    if (pdlFile.empty()) {
      return;
    }

    auto &ctx = this->getContext();
    auto op = getOperation();

    // Set up the input file.
    std::string errorMessage;
    auto file = openInputFile(pdlFile, &errorMessage);
    if (!file) {
      op->emitError("failed to load pdl file ") << pdlFile;
      signalPassFailure();
      return;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    auto pdlModule = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
    if (!pdlModule) {
      op->emitError("failed to parse pdl module ") << pdlFile;
      signalPassFailure();
      return;
    }

    PDLPatternModule pdlPattern(std::move(pdlModule));
    applyPDLPatternHooks(pdlPattern);
    RewritePatternSet patterns(&ctx);
    patterns.add(std::move(pdlPattern));
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createApplyPDLPatternsPass(llvm::StringRef pdlFile) {
  return std::make_unique<ApplyPDLPatternsPass>(pdlFile.str());
}
