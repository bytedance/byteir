//===- MhloToAce.cpp ------------------------------------------------------===//
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

#include "byteir/Conversion/ToAce/MhloToAce.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::ace;

namespace {

#include "MhloToAceActivationPattern.inc"

void populateFuseMhloToAceActivationPatterns(MLIRContext *context,
                                             RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

struct ConvertMhloToAcePass
    : public ConvertMhloToAceBase<ConvertMhloToAcePass> {
  void runOnOperation() override {
    func::FuncOp op = getOperation();
    MLIRContext *context = op.getContext();
    RewritePatternSet patterns(context);
    populateFuseMhloToAceActivationPatterns(context, patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(op, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertMhloToAcePass() {
  return std::make_unique<ConvertMhloToAcePass>();
}
