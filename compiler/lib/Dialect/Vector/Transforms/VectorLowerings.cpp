//===- VectorLowerings.cpp ----------------------------------*--- C++ -*-= == //
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

#include "byteir/Dialect/Vector/Transforms/Passes.h"

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_VECTORTRANSPOSELOWERINGPASS
#include "byteir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "vector-lowerings"

using namespace mlir;

namespace {

struct VectorTransposeLoweringPass
    : public impl::VectorTransposeLoweringPassBase<
          VectorTransposeLoweringPass> {
  using VectorTransposeLoweringPassBase::VectorTransposeLoweringPassBase;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    vector::populateVectorTransposeLoweringPatterns(
        patterns, vector::VectorTransformsOptions().setVectorTransposeLowering(
                      vector::VectorTransposeLowering::Shuffle));

    if (enableAVX2.getValue()) {
      x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
          patterns,
          x86vector::avx2::LoweringOptions().setTransposeOptions(
              x86vector::avx2::TransposeLoweringOptions()
                  .lower4x8xf32()
                  .lower8x8xf32()),
          /*benefit=*/100);
    }

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
