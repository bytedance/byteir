//===- FuseHloToCat.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/HloToCat/FuseHloToCat.h"
#include "byteir/Dialect/Cat/IR/CatDialect.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"
#include "./Utils.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace mlir {

namespace {

mlir::StringAttr
GetLayoutFrom3DDotGeneralDimNums(mlir::mhlo::DotDimensionNumbersAttr dims,
                                 Builder *builder) {
  auto ldims = dims.getLhsContractingDimensions();
  auto rdims = dims.getRhsContractingDimensions();
  assert(ldims.size() == 1 && rdims.size() == 1);
  if (ldims[0] == 2 && rdims[0] == 1)
    return builder->getStringAttr("rrr");
  if (ldims[0] == 2 && rdims[0] == 2)
    return builder->getStringAttr("rcr");
  if (ldims[0] == 1 && rdims[0] == 1)
    return builder->getStringAttr("crr");
  if (ldims[0] == 1 && rdims[0] == 2)
    return builder->getStringAttr("ccr");
}

mlir::StringAttr
GetLayoutFromConvDimNums(mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers,
                         Builder *builder) {
  auto layoutStr = getConvLayoutString(dimension_numbers);
  return builder->getStringAttr(layoutStr);
}

#include "FuseHloToCatPattern.inc"

struct FuseMhloToCatPass : public FuseMhloToCatBase<FuseMhloToCatPass> {
public:
  FuseMhloToCatPass() = default;
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    // ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateFuseMhloToCatPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createFuseMhloToCatPass() {
  return std::make_unique<FuseMhloToCatPass>();
}

void populateFuseMhloToCatPattern(RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

} // namespace mlir
