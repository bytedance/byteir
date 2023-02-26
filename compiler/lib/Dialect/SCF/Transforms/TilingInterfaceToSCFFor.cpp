//===- TilingInterfaceToSCFFor.cpp --------------------------------- C++ --===//
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
// Some code comes from TestTilingInterface.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/SCF/Transforms/TilingInterfaceToSCFFor.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;

namespace {
struct TilingInterfaceToSCFFor
    : public OpInterfaceRewritePattern<TilingInterface> {
  TilingInterfaceToSCFFor(MLIRContext *context,
                          std::function<bool(Operation *)> filter)
      : OpInterfaceRewritePattern(context), opFilter(filter) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (!opFilter(op))
      return rewriter.notifyMatchFailure(op, "was filtered out");
    FailureOr<SmallVector<scf::ForOp>> loops =
        scf::lowerToLoopsUsingSCFForOp(rewriter, op);
    if (failed(loops))
      return rewriter.notifyMatchFailure(op, "failed to lower to loops");
    rewriter.eraseOp(op);
    return loops;
  }

  std::function<bool(Operation *)> opFilter;
};
} // namespace

void mlir::scf::populateTilingInterfaceToSCFForPattern(
    mlir::RewritePatternSet &patterns,
    std::function<bool(Operation *)> opFilter) {
  patterns.add<TilingInterfaceToSCFFor>(patterns.getContext(), opFilter);
}
