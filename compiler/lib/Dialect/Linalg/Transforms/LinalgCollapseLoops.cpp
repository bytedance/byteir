//===- LinalgCollapseLoops.cpp --------------------------------*--- C++ -*-===//
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
// Some code comes from CollapseDimensions.cpp in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code comes from ElementwiseOpFusion.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/LinalgCollapseLoops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-collapse-loops"

namespace mlir {
#define GEN_PASS_DEF_LINALGCOLLAPSELOOPS
#include "byteir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

namespace {
/// Searches the same sequence in all the affine maps and collapses these
/// dimensions. It only applies these to "parallel" loops without mixing them
/// with "reduction" types.
static SmallVector<ReassociationIndices>
getCollapsibleLoops(linalg::GenericOp genericOp) {
  SmallVector<ReassociationIndices> contiguousLoops;

  SmallVector<unsigned> pDims;
  genericOp.getParallelDims(pDims);
  if (pDims.size() < 2)
    return contiguousLoops;

  llvm::SmallDenseSet<unsigned> pLoops(pDims.begin(), pDims.end());

  auto hasAllMapsSameSequence = [&](AffineExpr preExpr, AffineExpr nextExpr) {
    for (AffineMap map : genericOp.getIndexingMapsArray()) {
      bool foundSeq = false;
      for (auto [index, resultExpr] : llvm::enumerate(map.getResults())) {
        if (resultExpr == nextExpr) {
          foundSeq = (index > 0 && preExpr == map.getResult(index - 1));
          break;
        }
      }
      if (!foundSeq)
        return false;
    }
    return true;
  };

  ReassociationIndices range;
  AffineExpr preExpr;
  for (auto nextExpr : genericOp.getIndexingMapsArray().front().getResults()) {
    unsigned pos = nextExpr.cast<AffineDimExpr>().getPosition();
    if (!range.empty()) {
      if (!hasAllMapsSameSequence(preExpr, nextExpr) || !pLoops.count(pos)) {
        if (range.size() > 1)
          contiguousLoops.push_back({range.begin(), range.end()});
        range.clear();
      }
    }
    preExpr = nextExpr;
    if (pLoops.count(pos))
      range.push_back(pos);
  }
  if (range.size() > 1)
    contiguousLoops.push_back(range);

  LLVM_DEBUG({
    llvm::dbgs() << "Collapsing dimensions if possible: ";
    for (auto indices : contiguousLoops) {
      llvm::dbgs() << "[";
      for (auto idx : indices)
        llvm::dbgs() << idx << ",";
      llvm::dbgs() << "]\t";
    }
    llvm::dbgs() << "\n";
  });

  return contiguousLoops;
}

/// Returns true if the given op is collapsable.
static bool isEligibleForCollapse(linalg::GenericOp genericOp) {
  // TODO(guray) There is no mechanism to tell the collapsed indexes to
  // `tensor.expand_shape`. Once we have this support in MLIR, we can enable
  // dynamic tensor shapes.
  if (genericOp.hasDynamicShape())
    return false;

  // TODO(guray) Currently we can only collapse when result of all the
  // AffineMaps are dimensions. Possible to collapse cases like
  // affine_map<d0, d1+d2> with affine_map<d0, d1+d2>, however, this is not
  // supported in collapsing mechanism in MLIR. Once we have this support,
  // we can remove this if statement.
  if (llvm::any_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
        return !map.isProjectedPermutation();
      })) {
    return false;
  }

  // TODO(guray) Collapsing caused performance regression in a cpu
  // benchmark, so we disable it.
  if (genericOp.hasIndexSemantics())
    return false;

  if (!llvm::all_of(genericOp->getOperandTypes(), [](Type t) {
        if (auto memrefType = t.dyn_cast_or_null<MemRefType>()) {
          return memrefType.getLayout().isIdentity();
        }
        return true;
      })) {
    return false;
  }

  return true;
}
namespace {
class CollapsingInfo {
public:
  LogicalResult initialize(unsigned origNumLoops,
                           ArrayRef<ReassociationIndices> foldedIterationDims) {
    llvm::SmallDenseSet<int64_t, 4> processedDims;
    // Find all the dims that are folded.
    for (ReassociationIndicesRef foldedIterationDim : foldedIterationDims) {
      if (foldedIterationDim.empty())
        continue;
      // If the folded dims contain dims already folded, that's illegal
      // specification. Repetition within a list is also illegal.
      for (auto dim : foldedIterationDim) {
        if (dim >= origNumLoops)
          return failure();
        if (processedDims.count(dim))
          return failure();
        processedDims.insert(dim);
      }
      collapsedOpToOrigOpIterationDim.emplace_back(foldedIterationDim.begin(),
                                                   foldedIterationDim.end());
    }
    if (processedDims.size() > origNumLoops)
      return failure();

    // Add all the preserved dims of the original op as single
    // elements to `collapsedOpToOrigOpIterationDim`.
    for (auto dim : llvm::seq<int64_t>(0, origNumLoops)) {
      if (processedDims.count(dim))
        continue;
      collapsedOpToOrigOpIterationDim.emplace_back(ReassociationIndices{dim});
    }

    llvm::sort(collapsedOpToOrigOpIterationDim,
               [&](ReassociationIndicesRef lhs, ReassociationIndicesRef rhs) {
                 return lhs[0] < rhs[0];
               });
    origOpToCollapsedOpIterationDim.resize(origNumLoops);
    for (const auto &foldedDims :
         llvm::enumerate(collapsedOpToOrigOpIterationDim)) {
      for (const auto &dim : enumerate(foldedDims.value()))
        origOpToCollapsedOpIterationDim[dim.value()] =
            std::make_pair<int64_t, unsigned>(foldedDims.index(), dim.index());
    }
    return success();
  }

  /// Return mapping from collapsed loop domain to original loop domain.
  ArrayRef<ReassociationIndices> getCollapsedOpToOrigOpMapping() const {
    return collapsedOpToOrigOpIterationDim;
  }

  /// Return mapping from original loop domain to collapsed loop domain. The
  /// mapping is a pair. First value is the dimension in the collapsed loop that
  /// the original loop is mapped to. Second is the relative position in folded
  /// list of this domain. For example if the original loop domain is 3D, and
  /// the collapsed loop domain is folding all of it, i.e.
  ///
  /// ```
  /// collapsedOpToOrigOpMapping = [[0, 1, 2] [3, 4]]`
  /// ```
  ///
  /// then
  ///
  /// ```
  ///  origOpToCollapsedOpMapping[0] = {0, 0};
  ///  origOpToCollapsedOpMapping[1] = {0, 1};
  ///  origOpToCollapsedOpMapping[2] = {0, 2};
  ///  origOpToCollapsedOpMapping[3] = {1, 0};
  ///  origOpToCollapsedOpMapping[4] = {1, 1};
  /// ```
  ///
  ArrayRef<std::pair<int64_t, unsigned>> getOrigOpToCollapsedOpMapping() const {
    return origOpToCollapsedOpIterationDim;
  }

  /// Return the collapsed op iteration domain rank.
  unsigned getCollapsedOpIterationRank() const {
    return collapsedOpToOrigOpIterationDim.size();
  }

private:
  /// Map from the iteration domain index in collapsed op to the iteration
  /// domain indices in the original op.
  SmallVector<ReassociationIndices> collapsedOpToOrigOpIterationDim;

  /// Map from iteration domain index in the original op to the iteration domain
  /// index in the collapsed op.
  SmallVector<std::pair<int64_t, unsigned>> origOpToCollapsedOpIterationDim;
};
} // namespace

/// Get the iterator types for the collapsed operation given the original
/// iterator types and collapsed dimensions.
static SmallVector<utils::IteratorType>
getCollapsedOpIteratorTypes(ArrayRef<utils::IteratorType> iteratorTypes,
                            const CollapsingInfo &collapsingInfo) {
  SmallVector<utils::IteratorType> collapsedIteratorTypes;
  for (ReassociationIndicesRef foldedIterDims :
       collapsingInfo.getCollapsedOpToOrigOpMapping()) {
    assert(!foldedIterDims.empty() &&
           "reassociation indices expected to have non-empty sets");
    // Just pick the iterator type of the first folded dim. Pre-condition checks
    // expected to have checked that iterator types of all folded dimensions are
    // the same.
    collapsedIteratorTypes.push_back(iteratorTypes[foldedIterDims[0]]);
  }
  return collapsedIteratorTypes;
}

/// Compute the indexing map in the collapsed op that corresponds to the given
/// `indexingMap` of the original operation.
static AffineMap
getCollapsedOpIndexingMap(AffineMap indexingMap,
                          const CollapsingInfo &collapsingInfo) {
  MLIRContext *context = indexingMap.getContext();
  assert(indexingMap.isProjectedPermutation() &&
         "expected indexing map to be projected permutation");
  SmallVector<AffineExpr> resultExprs;
  auto origOpToCollapsedOpMapping =
      collapsingInfo.getOrigOpToCollapsedOpMapping();
  for (auto expr : indexingMap.getResults()) {
    unsigned dim = expr.cast<AffineDimExpr>().getPosition();
    // If the dim is not the first of the collapsed dim, do nothing.
    if (origOpToCollapsedOpMapping[dim].second != 0)
      continue;
    // The next n-dims are guaranteed to be collapsed. So just use the
    // iteration dimension of the collapsed op.
    resultExprs.push_back(
        getAffineDimExpr(origOpToCollapsedOpMapping[dim].first, context));
  }
  return AffineMap::get(collapsingInfo.getCollapsedOpIterationRank(), 0,
                        resultExprs, context);
}

/// Return the `reassociation` indices to use to collapse the operand when the
/// iteration space of a generic op is collapsed.
static SmallVector<ReassociationIndices>
getOperandReassociation(AffineMap indexingMap,
                        const CollapsingInfo &collapsingInfo) {
  unsigned counter = 0;
  SmallVector<ReassociationIndices> operandReassociation;
  auto origOpToCollapsedOpMapping =
      collapsingInfo.getOrigOpToCollapsedOpMapping();
  auto collapsedOpToOrigOpMapping =
      collapsingInfo.getCollapsedOpToOrigOpMapping();
  while (counter < indexingMap.getNumResults()) {
    unsigned dim =
        indexingMap.getResult(counter).cast<AffineDimExpr>().getPosition();
    // This is the start of a collapsed dimensions of the iteration that
    // is gauranteed to be preserved in the indexing map. The number of folded
    // dims is obtained from the collapsed op to original op mapping.
    unsigned numFoldedDims =
        collapsedOpToOrigOpMapping[origOpToCollapsedOpMapping[dim].first]
            .size();
    if (origOpToCollapsedOpMapping[dim].second == 0) {
      auto range = llvm::seq<unsigned>(counter, counter + numFoldedDims);
      operandReassociation.emplace_back(range.begin(), range.end());
    }
    counter += numFoldedDims;
  }
  return operandReassociation;
}

/// Get the new value to use for a given `OpOperand` in the collapsed operation.
static Value getCollapsedOpOperand(Location loc, GenericOp genericOp,
                                   OpOperand *opOperand,
                                   const CollapsingInfo &collapsingInfo,
                                   OpBuilder &builder) {
  AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
  SmallVector<ReassociationIndices> operandReassociation =
      getOperandReassociation(indexingMap, collapsingInfo);

  // If the number of entries in the reassocation for the operand is same as the
  // number of results of the indexing map, then nothing to do for this operand.
  Value operand = opOperand->get();
  if (operandReassociation.size() == indexingMap.getNumResults())
    return operand;

  // Insert a reshape to collapse the dimensions.
  if (genericOp.hasBufferSemantics()) {
    auto reshapeOp = builder.create<memref::CollapseShapeOp>(
        loc, operand, operandReassociation);
    return reshapeOp.getResult();
  } else {
    auto reshapeOp = builder.create<tensor::CollapseShapeOp>(
        loc, operand, operandReassociation);
    return reshapeOp.getResult();
  }
}

/// Modify the `linalg.index` operations in the original generic op, to its
/// value in the collapsed operation.
void generateCollapsedIndexingRegion(Location loc, Block *block,
                                     const CollapsingInfo &collapsingInfo,
                                     ValueRange loopRange,
                                     RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(block);

  // Collect all the original index ops.
  auto indexOps = llvm::to_vector(block->getOps<linalg::IndexOp>());

  // For each folded dimension list resolve the original induction variable
  // values in terms of the folded dimension induction variable.
  //   i_{folded} = (i_0 * d1 + i1) * d2 + i2.
  // can be inverted to
  //   i2 = i_{folded} % d2
  //   i1 = (i_{folded} / d2) % d1
  //   i0 = i_{folded} / (d1 * d2)
  llvm::DenseMap<unsigned, Value> indexReplacementVals;
  for (auto &foldedDims :
       enumerate(collapsingInfo.getCollapsedOpToOrigOpMapping())) {
    ReassociationIndicesRef foldedDimsRef(foldedDims.value());
    Value newIndexVal =
        rewriter.create<linalg::IndexOp>(loc, foldedDims.index());
    for (auto dim : llvm::reverse(foldedDimsRef.drop_front())) {
      indexReplacementVals[dim] =
          rewriter.create<arith::RemUIOp>(loc, newIndexVal, loopRange[dim]);
      newIndexVal =
          rewriter.create<arith::DivUIOp>(loc, newIndexVal, loopRange[dim]);
    }
    indexReplacementVals[foldedDims.value().front()] = newIndexVal;
  }

  for (auto indexOp : indexOps) {
    auto dim = indexOp.getDim();
    rewriter.replaceOp(indexOp, indexReplacementVals[dim]);
  }
}

/// Implementation of fusion with reshape operation by collapsing dimensions.
FailureOr<SmallVector<Value>> collapseGenericOpIterationDimsEx(
    GenericOp genericOp, ArrayRef<ReassociationIndices> foldedIterationDims,
    RewriterBase &rewriter) {
  // Bail on trivial no-op cases.
  if (genericOp.getNumLoops() <= 1 || foldedIterationDims.empty() ||
      llvm::all_of(foldedIterationDims, [](ReassociationIndicesRef foldedDims) {
        return foldedDims.size() <= 1;
      }))
    return failure();

  CollapsingInfo collapsingInfo;
  if (failed(collapsingInfo.initialize(genericOp.getNumLoops(),
                                       foldedIterationDims))) {
    return rewriter.notifyMatchFailure(
        genericOp, "illegal to collapse specified dimensions");
  }

  // Bail on non-canonical ranges.
  SmallVector<Range> loopRanges =
      cast<LinalgOp>(genericOp.getOperation())
          .createLoopRanges(rewriter, genericOp.getLoc());
  auto opFoldIsConstantValue = [](OpFoldResult ofr, int64_t value) {
    if (auto attr = ofr.dyn_cast<Attribute>())
      return attr.cast<IntegerAttr>().getInt() == value;
    llvm::APInt actual;
    return matchPattern(ofr.get<Value>(), m_ConstantInt(&actual)) &&
           actual.getSExtValue() == value;
  };
  if (!llvm::all_of(loopRanges, [&](Range range) {
        return opFoldIsConstantValue(range.offset, 0) &&
               opFoldIsConstantValue(range.stride, 1);
      })) {
    return rewriter.notifyMatchFailure(
        genericOp,
        "expected all loop ranges to have zero start and unit stride");
  }

  // Get the iterator types for the operand.
  SmallVector<utils::IteratorType> iteratorTypes = getCollapsedOpIteratorTypes(
      genericOp.getIteratorTypesArray(), collapsingInfo);

  // Get the indexing maps.
  auto indexingMaps = llvm::to_vector(
      llvm::map_range(genericOp.getIndexingMapsArray(), [&](AffineMap map) {
        return getCollapsedOpIndexingMap(map, collapsingInfo);
      }));

  Location loc = genericOp->getLoc();

  // Get the input operands.
  auto inputOperands = llvm::to_vector(llvm::map_range(
      genericOp.getDpsInputOperands(), [&](OpOperand *opOperand) {
        return getCollapsedOpOperand(loc, genericOp, opOperand, collapsingInfo,
                                     rewriter);
      }));

  // Get the output operands and result types.
  SmallVector<Type> resultTypes;
  SmallVector<Value> outputOperands;
  resultTypes.reserve(genericOp.getNumDpsInits());
  outputOperands.reserve(genericOp.getNumDpsInits());
  for (OpOperand *output : genericOp.getDpsInitOperands()) {
    Value newOutput =
        getCollapsedOpOperand(loc, genericOp, output, collapsingInfo, rewriter);
    outputOperands.push_back(newOutput);
    resultTypes.push_back(newOutput.getType());
  }

  // Create the generic op.
  linalg::GenericOp collapsedGenericOp;
  if (genericOp.hasBufferSemantics()) {
    collapsedGenericOp = rewriter.create<linalg::GenericOp>(
        loc, inputOperands, outputOperands, indexingMaps, iteratorTypes,
        [](OpBuilder &builder, Location loc, ValueRange args) {});
  } else {
    collapsedGenericOp = rewriter.create<linalg::GenericOp>(
        loc, resultTypes, inputOperands, outputOperands, indexingMaps,
        iteratorTypes,
        [](OpBuilder &builder, Location loc, ValueRange args) {});
  }
  Block *origOpBlock = &genericOp->getRegion(0).front();
  Block *collapsedOpBlock = &collapsedGenericOp->getRegion(0).front();
  rewriter.mergeBlocks(origOpBlock, collapsedOpBlock,
                       collapsedOpBlock->getArguments());

  if (collapsedGenericOp.hasIndexSemantics()) {
    // Collect the loop range of the generic op.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(collapsedGenericOp);
    SmallVector<Value> loopBound =
        llvm::to_vector(llvm::map_range(loopRanges, [&](Range range) {
          return getValueOrCreateConstantIndexOp(rewriter, loc, range.size);
        }));
    generateCollapsedIndexingRegion(loc,
                                    &collapsedGenericOp->getRegion(0).front(),
                                    collapsingInfo, loopBound, rewriter);
  }

  // Insert expanding reshape for the result to get back the original result
  // type.
  SmallVector<Value> results;
  for (const auto &originalResult : llvm::enumerate(genericOp->getResults())) {
    Value collapsedOpResult =
        collapsedGenericOp->getResult(originalResult.index());
    auto originalResultType =
        originalResult.value().getType().cast<ShapedType>();
    auto collapsedOpResultType = collapsedOpResult.getType().cast<ShapedType>();
    if (collapsedOpResultType.getRank() != originalResultType.getRank()) {
      AffineMap indexingMap =
          genericOp.getIndexingMapMatchingResult(originalResult.value());
      SmallVector<ReassociationIndices> reassociation =
          getOperandReassociation(indexingMap, collapsingInfo);
      if (genericOp.hasBufferSemantics()) {
        Value result = rewriter.create<memref::ExpandShapeOp>(
            loc, originalResultType, collapsedOpResult, reassociation);
        results.push_back(result);
      } else {
        Value result = rewriter.create<tensor::ExpandShapeOp>(
            loc, originalResultType, collapsedOpResult, reassociation);
        results.push_back(result);
      }
    } else {
      results.push_back(collapsedOpResult);
    }
  }
  return results;
}

class CollapseLoopsOnGenericOp : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Collect collapsible loops
    // TODO: All rules come from iree project, add our own
    if (!isEligibleForCollapse(op))
      return failure();
    auto loops = getCollapsibleLoops(op);
    if (loops.empty())
      return failure();

    // `collapseGenericOpIterationDimsEx` is similar to
    // `collapseGenericOpIterationDims` in upstream but allow buffer semantics
    // additionally
    Optional<SmallVector<Value>> replacements =
        collapseGenericOpIterationDimsEx(op, loops, rewriter);

    if (!replacements) {
      return failure();
    }

    rewriter.replaceOp(op, *replacements);
    return success();
  }
};

struct LinalgCollapseLoopsPass
    : public impl::LinalgCollapseLoopsBase<LinalgCollapseLoopsPass> {
  void runOnOperation() override {
    auto op = getOperation();
    auto context = op->getContext();

    RewritePatternSet patterns(context);
    patterns.add<CollapseLoopsOnGenericOp>(context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgCollapseLoops() {
  return std::make_unique<LinalgCollapseLoopsPass>();
}
