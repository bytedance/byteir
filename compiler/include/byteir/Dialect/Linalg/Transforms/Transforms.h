//===- Transforms.h -------------------------------------------*--- C++ -*-===//
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
// Some code comes from LinalgExt/Transforms/Transforms.h in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code comes from SCF/Transforms/TileUsingInterface.h in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H

#include "byteir/Utils/TileUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class DominanceInfo;
class PostDominanceInfo;

using TileFuncType = std::function<LogicalResult(
    RewriterBase &rewriter, TilingInterface op, ArrayRef<OpFoldResult> tileNums,
    ArrayRef<int64_t> interchange, ArrayRef<bool> useDistributdStyle,
    scf::SCFTileAndFuseResult &tileAndFuseResult)>;

namespace scf {
/// tileConsumerAndFuseProducerUsingSCFForOpExt is an enhanced version
/// tileConsumerAndFuseProducerGreedilyUsingSCFForOp.
FailureOr<scf::SCFTileAndFuseResult>
tileConsumerAndFuseProducerUsingSCFForOpExt(
    RewriterBase &rewriter, TilingInterface consumer,
    ArrayRef<Operation *> stopOps, const scf::SCFTileAndFuseOptions &options,
    bool simplifyLoopIter = true, bool keepIntermediate = false);

/// @brief  This is an enhancement version of upstream's
/// tileConsumerAndFuseProducerGreedilyUsingSCFFor to tile & fuse multiple root
/// @param tensors the roots of the tile & fuse procedure
/// @param expectWholeGraphFusion if set True, return failure() if a whole graph
/// tile & fuse cannot be performed
FailureOr<scf::SCFTileAndFuseResult>
tileConsumerArrayAndFuseProducerGreedilyUsingSCFFor(
    RewriterBase &rewriter, ArrayRef<Value> tensors,
    const TilingOptions &options, TileFuncType tileFunc = nullptr,
    bool expectWholeGraphFusion = false);

void labelTileLoopType(Operation *op, ArrayRef<scf::ForOp> loops);

LogicalResult isValidTiling(Operation *tiled);

LogicalResult isValidFusibleProducerOp(OpOperand &consumer,
                                       Operation *fusibleProducerOp);

bool isResultLoopInvariant(Operation *op, int64_t resultNumber,
                           bool hasOneOrZeroUse, bool allParallel);

} // namespace scf

namespace linalg {

bool isProducerElementwiseOpFusable(OpOperand *consumerOpOperand);

/// Rewrite a fusion pattern of an elementwise consumer with elementwise
/// producers
void populateElementwiseOpsProducerConsumerFusionPatterns(
    RewritePatternSet &patterns, bool diffShape,
    const linalg::ControlFusionFn &controlElementwiseOpFusion,
    DominanceInfo &dom, PostDominanceInfo &post);

/// Rewrite linalg::MapOp to linalg::GenericOp
void populateMapOpToGenericPattern(RewritePatternSet &patterns);

} // namespace linalg

namespace linalg_ext {

/// Insert linalg_ext::AliasOp for a shared input to help fusion
void populateInsertLinalgExtAliasForSharedInputFusionPatterns(
    RewritePatternSet &patterns, DominanceInfo &dom);

/// Remove linalg_ext::AliasOp by replacing it with its input
void populateRemoveLinalgExtAliasPattern(RewritePatternSet &patterns);

/// return a list of utils::IteratorType for a given op
/// and list of scf::ForOp loops
///
/// ```mlir
/// Example 1:
/// scf.for %iv_m      // m_loop
///   scf.for %iv_k    // k_loop
///     scf.for %iv_n  // n_loop
///       extract_slice_A
///       extract_slice_B
///       extract_slice_C
///       %0 = linalg.matmul ins (extract_slice_A, extract_slice_B)
///                          outs(extract_slice_C)
/// ```
/// loops = [m_loop, k_loop, n_loop], op = linalg.matmul
/// return [parallel, reduction, parallel]
///
FailureOr<llvm::SmallVector<std::optional<utils::IteratorType>>>
getLoopIteratorTypes(Operation *op, ArrayRef<scf::ForOp> loops);

void mergeLoopIteratorTypes(
    llvm::SmallVector<std::optional<utils::IteratorType>> &from,
    llvm::SmallVector<std::optional<utils::IteratorType>> &to);

// Simplify a dimOp of linalg and linalg-ext
LogicalResult simplifyTensorDimOpUsedInLinalg(RewriterBase &rewriter,
                                              Operation *op);

void simplifyTensorDimOpUsedInLinalgWithinOp(Operation &op);

// LinalgTransforms and LinalgTransformationFilter will be deprecated soon
struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

/// Helper class to control application of linalg transformation patterns.
/// Control comes in 2 forms:
///   1. attribute matching and setting behavior using the attribute named
///      `kLinalgTransformMarker`. This can be used to build a state machine
///      using attributes and incrementally applying patterns to advance states.
///   2. filter function, which is a simple lambda on the Operation* that
///      returns a LogicalResult.
struct LinalgTransformationFilter {
  using FilterFunction = std::function<LogicalResult(Operation *)>;

  explicit LinalgTransformationFilter(
      ArrayRef<StringAttr> matchDisjunction = {},
      std::optional<StringAttr> replacement = std::nullopt);

  explicit LinalgTransformationFilter(
      const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction = {},
      std::optional<StringAttr> replacement = std::nullopt);

  LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
  LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                         Operation *op) const;
  bool hasReplacementFilter(Operation *op) const;

  LinalgTransformationFilter &addFilter(const FilterFunction &f) {
    if (f)
      filters.push_back(f);
    return *this;
  }

  template <typename... OpTypes> LinalgTransformationFilter &addOpFilter() {
    return addFilter(
        [](Operation *op) { return success(isa<OpTypes...>(op)); });
  }

  LinalgTransformationFilter &addOpNameFilter(StringRef opName) {
    return addFilter([opName](Operation *op) {
      return success(op->getName().getStringRef() == opName);
    });
  }

  LinalgTransformationFilter &setMatchByDefault() {
    matchByDefault = true;
    return *this;
  }

private:
  SmallVector<FilterFunction> filters;
  SmallVector<StringAttr> matchDisjunction;
  std::optional<StringAttr> replacement;
  /// When set to true, if the attribute is not set, it will be treated as
  /// a match. Default is false.
  bool matchByDefault;
};

} // namespace linalg_ext
} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TRANSFORMS_H
