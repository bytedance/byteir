//===- CanonicalizeExt.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Tensor/Transforms/CanonicalizeExt.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

#define DEBUG_TYPE "tensor-canonicalize-ext"

#define K_INITIAL -999

using namespace mlir;

void mlir::tensor::populateCanonicalizeExtPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx,
                                                   bool blindFold) {
  if (blindFold) {
    populateFoldConstantExtractSlicePatterns(
        patterns, [](ExtractSliceOp op) { return true; });
  }
}

void mlir::tensor::getCanonicalizationExtPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx,
                                                  bool blindFold) {

  // add dialect level getCanonicalizationPatterns
  auto tensorDialect = ctx->getOrLoadDialect<tensor::TensorDialect>();
  if (tensorDialect) {
    tensorDialect->getCanonicalizationPatterns(patterns);
  }

  // add op level  getCanonicalizationPatterns
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    // only add tensor-related
    if (isa<tensor::TensorDialect>(op.getDialect())) {
      op.getCanonicalizationPatterns(patterns, ctx);
    }
  }

  // add our extension
  populateCanonicalizeExtPatterns(patterns, ctx, blindFold);
}
