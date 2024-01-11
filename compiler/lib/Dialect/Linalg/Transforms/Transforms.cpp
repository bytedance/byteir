//===- Transforms.cpp -----------------------------------------*--- C++ -*-===//
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
// Some code comes from Tiling.cpp in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code comes from TileUsingInterface.cpp and Generalization.cpp
// in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/Transforms.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Util/Util.h"
#include "byteir/Dialect/SCF/Util/Util.h"
#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/GraphUtils.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-ext-transforms"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;
using namespace mlir::scf;

using IteratorTypes = llvm::SmallVector<std::optional<utils::IteratorType>>;

//===----------------------------------------------------------------------===//
// populateMapOpToGenericPattern
//===----------------------------------------------------------------------===//

namespace {

/// Patterns to rewrite a map a generic op
class MapOpToGenericOp : public OpRewritePattern<linalg::MapOp> {
public:
  MapOpToGenericOp(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MapOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg::MapOp mapOp,
                                PatternRewriter &rewriter) const override {
    // MapOp currently has no RegionBuilder,
    // so cannot directly call linalg::generalizeNamedOp
    // TODO: change code back to calling generalizeNamedOp,
    //       if upstream starting support MapOp's generalization.
    auto linalgOp = cast<linalg::LinalgOp>(mapOp.getOperation());
    SmallVector<Value> inputs = linalgOp.getDpsInputOperands();
    SmallVector<Value> outputs = linalgOp.getDpsInitOperands();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iterators =
        linalgOp.getIteratorTypesArray();
    SmallVector<Type> resultTypes = linalgOp.hasTensorSemantics()
                                        ? TypeRange(ValueRange(outputs))
                                        : TypeRange{};
    GenericOp genericOp =
        rewriter.create<GenericOp>(linalgOp.getLoc(), resultTypes, inputs,
                                   outputs, indexingMaps, iterators);

    // Inline mapOp's bb into genericOp
    rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                                genericOp.getRegion().begin());

    // Add output addArgument,
    // since genericOp's bb supports output argument, but mapOp's bb doesn't.
    auto block = genericOp.getBlock();
    auto loc = genericOp.getLoc();
    for (auto output : outputs) {
      block->addArgument(output.getType().cast<ShapedType>().getElementType(),
                         loc);
    }

    rewriter.replaceOp(linalgOp, genericOp->getResults());
    return success();
  }
};

} // namespace

void mlir::linalg::populateMapOpToGenericPattern(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<MapOpToGenericOp>(context);
}

//===----------------------------------------------------------------------===//
// mergeLoopIteratorTypes
//===----------------------------------------------------------------------===//

void mlir::linalg_ext::mergeLoopIteratorTypes(
    llvm::SmallVector<std::optional<utils::IteratorType>> &from,
    llvm::SmallVector<std::optional<utils::IteratorType>> &to) {
  // logic:
  // parallel, parallel => parallel
  // parallel, none => parallel
  // parallel, reduce => reduce
  // none, none => none
  // none, reduce => reduce
  // reduce, x => reduce
  for (const auto &en : llvm::enumerate(from)) {
    if (en.value().has_value()) {
      if (to[en.index()].has_value() && *en.value() != *to[en.index()]) {
        // when (iterTy, curTy) == (parallel, reduce) or (reduce, parallel)
        // assign iterTy = reduce
        to[en.index()] = utils::IteratorType::reduction;
      } else {
        // when either iterTy is none or iterTy == curTy
        // assign iterTy = curTy
        to[en.index()] = *en.value();
      }
    }
  } // for en : llvm::enumerate(from)
}

//===----------------------------------------------------------------------===//
// simplifyLinalgOp
//===----------------------------------------------------------------------===//

namespace {

/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

// return getIndexingMapsArray if an op having getIndexingMapsArray
FailureOr<llvm::SmallVector<AffineMap>> getIndexingMapsArray(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgOp.getIndexingMapsArray();
  } else if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return linalgExtOp.getIndexingMapsArray();
  }
  return failure();
}

LogicalResult
replaceTensorDim(RewriterBase &rewriter, tensor::DimOp dimOp, size_t offset,
                 AffineMap concatMap,
                 llvm::DenseMap<AffineExpr, std::tuple<Value, int64_t>>
                     &exprToTensorAndDim) {

  auto maybeConstIndex = dimOp.getConstantIndex();
  if (!maybeConstIndex.has_value())
    return failure();
  unsigned exprOffset = offset + *maybeConstIndex;
  auto affineExpr = concatMap.getResult(exprOffset);
  assert(exprToTensorAndDim.count(affineExpr) > 0);

  auto [source, dimIdx] = exprToTensorAndDim[affineExpr];

  // check whether it map to itself
  // if so, no need to replace
  if (source == dimOp.getShapedValue() && *maybeConstIndex == dimIdx) {
    return failure();
  }

  // create a new DimdOp
  rewriter.setInsertionPoint(dimOp);
  rewriter.replaceOpWithNewOp<tensor::DimOp>(dimOp, source, dimIdx);
  return success();
}
} // namespace

LogicalResult
mlir::linalg_ext::simplifyTensorDimOpUsedInLinalg(RewriterBase &rewriter,
                                                  Operation *op) {
  auto maybeIndexingMapArray = getIndexingMapsArray(op);
  if (failed(maybeIndexingMapArray)) {
    return failure();
  }

  AffineMap concatMap = concatAffineMaps(*maybeIndexingMapArray);
  DenseMap<AffineExpr, std::tuple<Value, int64_t>> exprToTensorAndDim;

  unsigned offset = 0;
  auto updateExprToTensorAndDim = [&](Value tensor, int64_t dim) {
    auto resultExpr = concatMap.getResult(offset);
    if (exprToTensorAndDim.count(resultExpr) == 0) {
      exprToTensorAndDim[resultExpr] = std::make_tuple(tensor, dim);
    }
  };

  // preprocessing Tensors
  if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (auto opOperand : dstOp.getDpsInputOperands()) {
      auto tensor = opOperand->get();
      if (auto shapeTy = tensor.getType().dyn_cast<ShapedType>()) {
        for (int64_t d = 0; d < shapeTy.getRank(); ++d) {
          updateExprToTensorAndDim(tensor, d);
          offset++;
        }
      }
    }

    for (auto opOperand : dstOp.getDpsInitOperands()) {
      auto tensor = opOperand->get();
      if (auto shapeTy = tensor.getType().dyn_cast<ShapedType>()) {
        for (auto d = 0; d < shapeTy.getRank(); ++d) {
          updateExprToTensorAndDim(tensor, d);
          offset++;
        }
      }
    }
  }

  offset = 0;
  bool isSucceeded = false;
  auto applyReplaceTensorDimAndUpdateOffset = [&](Value tensor) {
    if (auto shapeTy = tensor.getType().dyn_cast<ShapedType>()) {
      unsigned rank = shapeTy.getRank();
      for (auto user : llvm::make_early_inc_range(tensor.getUsers())) {
        if (auto dimOp = dyn_cast<tensor::DimOp>(user)) {
          if (succeeded(replaceTensorDim(rewriter, dimOp, offset, concatMap,
                                         exprToTensorAndDim))) {
            isSucceeded = true;
          }
        }
      }
      offset += rank;
    }
  };

  // now replace all
  if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (auto opOperand : dstOp.getDpsInputOperands()) {
      auto tensor = opOperand->get();
      applyReplaceTensorDimAndUpdateOffset(tensor);
    }

    unsigned inputOffset = offset;

    for (auto opOperand : dstOp.getDpsInitOperands()) {
      auto tensor = opOperand->get();
      applyReplaceTensorDimAndUpdateOffset(tensor);
    }

    offset = inputOffset;
    for (auto tensor : op->getResults()) {
      applyReplaceTensorDimAndUpdateOffset(tensor);
    }
  }

  return success(isSucceeded);
}

void mlir::linalg_ext::simplifyTensorDimOpUsedInLinalgWithinOp(Operation &op) {
  SimpleRewriter rewriter(op.getContext());
  SmallVector<Operation *> linalgOps;
  op.walk([&](Operation *inner) {
    if (isa<linalg::LinalgOp, linalg_ext::LinalgExtOp>(inner)) {
      linalgOps.push_back(inner);
    }
  });

  // reverse order
  for (auto it = linalgOps.rbegin(); it < linalgOps.rend(); ++it) {
    (void)simplifyTensorDimOpUsedInLinalg(rewriter, *it);
  }
}

//===----------------------------------------------------------------------===//
// labelTileLoopType
//===----------------------------------------------------------------------===//

void mlir::scf::labelTileLoopType(Operation *op, ArrayRef<scf::ForOp> loops) {
  if (op == nullptr) {
    return;
  }

  auto innerMostSCFFor = loops.back();
  if (innerMostSCFFor.getBody() != op->getBlock()) {
    return;
  }

  IteratorTypes iterTys(loops.size(), std::nullopt);

  for (auto &innerOp : innerMostSCFFor.getBody()->without_terminator()) {
    if (!isa<TilingInterface>(innerOp)) {
      continue;
    }

    FailureOr<IteratorTypes> curTys = getLoopIteratorTypes(&innerOp, loops);
    if (failed(curTys)) {
      continue;
    }

    // merge IteratorTypes to iterTys
    mergeLoopIteratorTypes(*curTys, iterTys);
  }

  auto ctx = op->getContext();
  for (const auto &en : llvm::enumerate(iterTys)) {
    if (en.value().has_value() &&
        *en.value() == utils::IteratorType::parallel) {
      loops[en.index()]->setAttr(getSCFForParallelAttrName(),
                                 UnitAttr::get(ctx));
    }
  }
}

//===----------------------------------------------------------------------===//
// isValidTiling
//===----------------------------------------------------------------------===//

LogicalResult mlir::scf::isValidTiling(Operation *tiled) {
  if (tiled == nullptr)
    return failure();

  if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(tiled)) {
    return linalgExtOp.isValidTiling(tiled);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// isResultLoopInvariant
//===----------------------------------------------------------------------===//

bool mlir::scf::isResultLoopInvariant(Operation *op, int64_t resultNumber,
                                      bool passUsesCheck, bool allParallel) {
  if (op == nullptr)
    return false;

  if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return linalgExtOp.isResultLoopInvariant(resultNumber, passUsesCheck,
                                             allParallel);
  } else if (isa<linalg::LinalgOp>(op)) {
    return passUsesCheck && allParallel;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// getLoopIteratorTypes
//===----------------------------------------------------------------------===//

namespace {

bool isNewValue(Value val) {
  if (auto def = val.getDefiningOp()) {
    return isa<tensor::EmptyOp>(def);
  }
  return false;
}

FailureOr<bool> getLocalComputation(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return llvm::all_of(linalgOp.getDpsInitOperands(), [&](OpOperand *opVal) {
      return isNewValue(opVal->get());
    });
  } else if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return llvm::all_of(linalgExtOp.getOutputOperands(), [&](OpOperand *opVal) {
      return isNewValue(opVal->get());
    });
  }
  return failure();
}

} // namespace

FailureOr<IteratorTypes>
mlir::linalg_ext::getLoopIteratorTypes(Operation *op,
                                       ArrayRef<scf::ForOp> loops) {

  // early termination if no TilingInterface
  if (!isa<TilingInterface>(op)) {
    return failure();
  }

  FailureOr<llvm::SmallVector<AffineMap>> indexingMaps =
      getIndexingMapsArray(op);
  // early termination if no indexingMaps
  // TODO: relax this, by making indexingMaps all reduce
  // TODO: support tensor::expand_shape or collapse_shape
  if (failed(indexingMaps)) {
    LLVM_DEBUG(DBGS() << "skip getLoopIteratorTypes due to no indexingMaps\n");
    return failure();
  }

  FailureOr<bool> localComputation = getLocalComputation(op);
  // early termination if no indexingMaps
  // TODO: relax this, by making localComputation false.
  if (failed(localComputation)) {
    LLVM_DEBUG(DBGS() << "skip getLoopIteratorTypes due to no support of " << op
                      << "\n");
    return failure();
  }

  auto tilingLoopIterType = cast<TilingInterface>(op).getLoopIteratorTypes();
  IteratorTypes retIterTys(loops.size(), std::nullopt);

  // preset LoopIV to loopIdx
  DenseMap<Value, size_t> loopIV2Idx;
  for (const auto &en : llvm::enumerate(loops)) {
    auto forOp = en.value();
    loopIV2Idx[forOp.getInductionVar()] = en.index();
  }

  // check all args
  for (const auto &en : llvm::enumerate(op->getOperands())) {
    llvm::SmallVector<::mlir::OpFoldResult, 4> mixedOffsets;

    IteratorTypes anotherIterTys;

    if (auto sliceOp = en.value().getDefiningOp<tensor::ExtractSliceOp>()) {
      mixedOffsets = sliceOp.getMixedOffsets();
    } else if (auto subviewOp = en.value().getDefiningOp<memref::SubViewOp>()) {
      mixedOffsets = subviewOp.getMixedOffsets();
    } else if (auto defOp = en.value().getDefiningOp()) {
      // only allow some op doing recursion
      if (!isa<linalg::BroadcastOp, linalg::TransposeOp, linalg::FillOp>(
              defOp)) {
        continue;
      }
      // handle recursive
      auto maybeIterTypes = getLoopIteratorTypes(defOp, loops);
      if (succeeded(maybeIterTypes)) {
        anotherIterTys = *maybeIterTypes;
        mergeLoopIteratorTypes(anotherIterTys, retIterTys);
      }
      continue;
    } else {
      continue;
    }

    auto indexingMap = (*indexingMaps)[en.index()];
    for (const auto &en2 : llvm::enumerate(mixedOffsets)) {
      Value argVal = en2.value().dyn_cast<Value>();

      if (!argVal) {
        // skip when argVal folded to a const
        // implying not a loop iv
        continue;
      }

      // handle apply case
      if (auto apply = argVal.getDefiningOp<affine::AffineApplyOp>()) {
        // supporting 1D case for now
        // TODO: extend to n-D cases
        if (apply.getAffineMap().getNumDims() == 1 &&
            apply.getMapOperands().size() == 1) {
          // update argVal to AffineApplyOp's input
          argVal = apply.getMapOperands()[0];
        }
      }

      if (loopIV2Idx.count(argVal) == 0) {
        // skip when not in loopIV2Idx
        // implying not a loop iv
        continue;
      }

      FailureOr<unsigned> iterAxis =
          getIterAxisFromDim(indexingMap, en2.index());
      if (failed(iterAxis)) {
        // skip when iterAxis not found
        continue;
      }

      auto iterTy = *localComputation ? utils::IteratorType::parallel
                                      : tilingLoopIterType[*iterAxis];
      auto loopIdx = loopIV2Idx[argVal];

      if (retIterTys[loopIdx].has_value()) {
        if (*retIterTys[loopIdx] != iterTy) {
          // detect more than one LoopIterType
          return failure();
        }
      } else {
        // if has no value, set it now
        retIterTys[loopIdx] = iterTy;
      }
    } // for en2 : llvm::enumerate(mixedOffsets)
  }   // for en : llvm::enumerate(op->getOperands()))

  return retIterTys;
}

//===----------------------------------------------------------------------===//
// isValidFusibleProducerOp
//===----------------------------------------------------------------------===//

LogicalResult scf::isValidFusibleProducerOp(OpOperand &consumer,
                                            Operation *fusibleProducerOp) {
  if (auto linalgExtConsumerOp = dyn_cast<LinalgExtOp>(consumer.getOwner())) {
    if (failed(linalgExtConsumerOp.isValidTiledProducerOp(
            fusibleProducerOp, consumer.getOperandNumber()))) {
      return failure();
    }
  } else if (auto linalgConsumerOp = dyn_cast<LinalgOp>(consumer.getOwner())) {
    auto tiledOp = cast<TilingInterface>(consumer.getOwner());
    if (involveReduction(*consumer.getOwner(),
                         linalgConsumerOp.getIndexingMapsArray(),
                         tiledOp.getLoopIteratorTypes())) {
      // if there is a reduction and init shouldn't be fused
      // [FIXME] (lwc) this might overkill
      if (consumer.getOperandNumber() >= linalgConsumerOp.getNumDpsInputs()) {
        return failure();
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tileConsumerAndFuseProducerUsingSCFForOpExt
//===----------------------------------------------------------------------===//

namespace {

// confirm whether a fusion valid through `makeValidTiledConsumerOps`, and
// `isValidFusibleProducerOp`
LogicalResult confirmValidFusion(OpBuilder &b, OpResult unFusedProducer,
                                 Operation *fusedProducerOp,
                                 tensor::ExtractSliceOp slice) {
  Operation *unFusedProducerOp = unFusedProducer.getOwner();
  if (auto linalgExtUnfused = dyn_cast<LinalgExtOp>(unFusedProducerOp)) {
    return linalgExtUnfused.makeValidTiledConsumerOps(
        b, fusedProducerOp, unFusedProducer.getResultNumber());
  }

  return success();
}

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the iter_args
/// indicates that this is a destination operand of the consumer. If there was
/// no loop traversal needed, the second value of the returned tuple is empty.
static std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<scf::ForOp> loops) {
  std::optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin(); // inner to outer
  while (auto iterArg = source->get().dyn_cast<BlockArgument>()) {
    scf::ForOp loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = &loop.getOpOperandForRegionIterArg(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend()) {
    destinationIterArg = source;
  }

  return {source->get().dyn_cast<OpResult>(), destinationIterArg};
}

/// If the tiled operation is destination passing style, update the
/// slice of the destination used (which refers to the untiled destination)
/// to use the corresponding region argument of the innermost loop.
///
/// ```mlir
/// %0 =
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %0
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
///
/// is transformed to
///
/// ```mlir
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %arg
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
static void
updateDestinationOperandsForTiledOp(OpBuilder &builder,
                                    ValueRange tiledOpDestinationValues,
                                    ValueRange bbArgsList) {
  for (const auto &destValue : llvm::enumerate(tiledOpDestinationValues)) {
    auto sliceOp = destValue.value().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      continue;
    sliceOp.setOperand(0, bbArgsList[destValue.index()]);
  }
}

// update replacements when oldLoops changing to newLoops
static void updateReplacements(llvm::DenseMap<Value, Value> &replacements,
                               ArrayRef<scf::ForOp> oldLoops,
                               ArrayRef<scf::ForOp> newLoops) {
  // generate loop map
  llvm::DenseMap<scf::ForOp, scf::ForOp> oldToNewLoop;
  for (const auto &en : llvm::enumerate(oldLoops)) {
    oldToNewLoop[en.value()] = newLoops[en.index()];
  }

  for (auto &it : replacements) {
    if (auto oldResult = dyn_cast<OpResult>(it.second)) {
      if (auto oldLoop = dyn_cast<scf::ForOp>(oldResult.getOwner())) {
        if (oldToNewLoop.count(oldLoop) > 0) {
          auto newResult =
              oldToNewLoop[oldLoop]->getResult(oldResult.getResultNumber());
          it.second = newResult;
        }
      }
    }
  }
}

/// For a value to be yielded (`yieldedValue`) from within a loop nest `loops`,
/// construct the destructive update pattern that inserts the yielded
/// value into a destination tensor provided by `initValue` at offset
/// `tileOffsets` and size `tileSizes`. For example,
///
/// ```mlir
/// scf.for %iv0 = ... {
///   %0 = tiled_op
/// }
/// ```
///
/// is transformed to
///
/// ```mlir
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %arg
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
/// TODO: This API can be cleaned up by using `SubsetExtractOpInterface`.
///
/// This function is modified by adding functionality of updating replacements
static LogicalResult
yieldTiledValues(RewriterBase &rewriter, ValueRange initValues,
                 ValueRange yieldedValues,
                 ArrayRef<SmallVector<OpFoldResult>> tileOffsetsList,
                 ArrayRef<SmallVector<OpFoldResult>> tileSizesList,
                 MutableArrayRef<scf::ForOp> loops,
                 llvm::DenseMap<Value, Value> &replacements,
                 std::optional<OpOperand *> &destinationIterArg) {
  NewYieldValueFn yieldValueFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBBArgs) -> SmallVector<Value> {
    SmallVector<Value> inserts;
    for (const auto &yieldedValue : llvm::enumerate(yieldedValues)) {
      ArrayRef<OpFoldResult> tileOffsets =
          tileOffsetsList[yieldedValue.index()];
      ArrayRef<OpFoldResult> tileSizes = tileSizesList[yieldedValue.index()];
      SmallVector<OpFoldResult> tileStrides(tileOffsets.size(),
                                            b.getIndexAttr(1));
      Value insert = b.create<tensor::InsertSliceOp>(
          loc, yieldedValue.value(), newBBArgs[yieldedValue.index()],
          tileOffsets, tileSizes, tileStrides);
      inserts.push_back(insert);
    }
    return inserts;
  };

  SmallVector<scf::ForOp> newLoops =
      replaceLoopNestWithNewYields(rewriter, loops, initValues, yieldValueFn,
                                   /*replaceIterOperandsUsesInLoop =*/false);

  // this functionality is added on top of the exisitng upstream version
  updateReplacements(replacements, loops, newLoops);

  // update destinationIterArg
  if (destinationIterArg.has_value()) {
    for (const auto &loop : llvm::enumerate(loops)) {
      // check old loop is the destinationIterArg's getOwner
      if ((*destinationIterArg)->getOwner() == loop.value()) {
        *destinationIterArg = &newLoops[loop.index()]->getOpOperand(
            (*destinationIterArg)->getOperandNumber());
      }
    }
  }

  // remove loops and make newLoops
  for (const auto &loop : llvm::enumerate(loops)) {
    rewriter.eraseOp(loop.value());
    loops[loop.index()] = newLoops[loop.index()];
  }
  return success();
}

// create insertSliceOp for results
static LogicalResult
createResultSlices(RewriterBase &rewriter, Operation *op, Operation *tiledOp,
                   tensor::ExtractSliceOp sliceOp,
                   SmallVector<scf::ForOp> &loops,
                   llvm::DenseMap<Value, Value> &replacements,
                   std::optional<OpOperand *> &destinationIterArg) {
  if (!isa<TilingInterface>(op)) {
    return failure();
  }

  SmallVector<Value> destinationTensors; // tensor before tiling.
  if (failed(tensor::getOrCreateDestinations(rewriter, op->getLoc(), op,
                                             destinationTensors))) {
    LLVM_DEBUG(DBGS() << "[createResultSlices] failed to get destinations for "
                      << *op << "\n");
    return rewriter.notifyMatchFailure(op, "failed to get destinations");
  }

  int64_t numResults = op->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsetsList(numResults),
      resultSizesList(numResults);
  auto outputRange = tiledOp->getOperands().take_back(numResults);
  for (const auto &result : llvm::enumerate(op->getResults())) {
    auto tiledOutput = outputRange[result.index()];
    if (auto sliceOp =
            dyn_cast<tensor::ExtractSliceOp>(tiledOutput.getDefiningOp())) {
      resultOffsetsList[result.index()] = sliceOp.getMixedOffsets();
      resultSizesList[result.index()] = sliceOp.getMixedSizes();
    } else {
      LLVM_DEBUG(
          DBGS()
          << "[createResultSlices] handle non-slice by creating a entire "
             "view\n");
      // TODO: handle non-slice by creating a entire view
      return failure();
    }
  }

  auto oldNumResult = loops.front()->getNumResults();
  if (failed(yieldTiledValues(rewriter, destinationTensors,
                              tiledOp->getResults(), resultOffsetsList,
                              resultSizesList, loops, replacements,
                              destinationIterArg))) {
    return rewriter.notifyMatchFailure(op, "failed to yield replacement");
  }

  if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(tiledOp)) {
    auto innerMostLoop = loops.back();
    SmallVector<Value> destinationTensors = dstOp.getDpsInitOperands();

    updateDestinationOperandsForTiledOp(
        rewriter, destinationTensors,
        innerMostLoop.getRegionIterArgs().take_back(destinationTensors.size()));
  }

  // update replacements
  for (const auto &en : llvm::enumerate(op->getResults())) {
    replacements[en.value()] =
        loops.front()->getResult(oldNumResult + en.index());
  }

  return success();
}

static void getProducerAndConsumerTensorSlices(
    Operation *op, llvm::DenseMap<Value, Value> &iterArgToOperand,
    SmallPtrSetImpl<Operation *> &opCollection,
    SmallPtrSetImpl<Value> &valCollection) {
  for (const auto val : op->getOperands()) {
    if (auto sliceOp = val.getDefiningOp<tensor::ExtractSliceOp>()) {
      // insert to opCollection
      if (!opCollection.contains(sliceOp))
        opCollection.insert(sliceOp);

      Value src = sliceOp.getSource();
      if (iterArgToOperand.count(src) > 0 && !valCollection.contains(src)) {
        valCollection.insert(src);
      }
    }
  }

  for (const auto val : op->getResults()) {
    for (const auto userOp : val.getUsers()) {
      if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(userOp)) {
        if (!opCollection.contains(userOp))
          opCollection.insert(userOp);
        Value dst = sliceOp.getDest();
        if (iterArgToOperand.count(dst) > 0 && !valCollection.contains(dst)) {
          valCollection.insert(dst);
        }
      }
    }
  }
}

/// For several tensor.extract_slice ops of the same source, merge all or part
/// of them who slice on the same group of dimensions and the same op result as
/// well. This is a must in models with residual blocks.
///             op0
///              |  \
///              |   conv0
///              |   /
///             op1
///              |  \
///              |   conv1
///              |   /
///             ...
///              |  \
///              |   convN
///              |   /
///             opN+1
/// For a model with N residual blocks, op0 has to be tiled by 2**N times if the
/// tensor.extract_slice ops are not merged.
///
/// Note: the topological order mighe be broken after this function
SmallVector<tensor::ExtractSliceOp>
mergeSliceOps(SmallVector<tensor::ExtractSliceOp> &sliceOps) {
  if (sliceOps.size() == 0)
    return {};

  // declare variables for later use
  SmallVector<tensor::ExtractSliceOp> mergedSliceOps;
  OpBuilder builder(sliceOps[0]);

  // group the slice ops according to their sliced dimensions
  auto getSliceKey = [&](tensor::ExtractSliceOp sliceOp) {
    ArrayRef<int64_t> offsets = sliceOp.getStaticOffsets();
    int64_t key = 0;
    int64_t multiplier = 1;
    for (int64_t offset : offsets) {
      if (ShapedType::isDynamic(offset))
        key += multiplier;
      multiplier = multiplier << 1;
    }
    return key;
  };
  mlir::DenseMap<int64_t, SmallVector<tensor::ExtractSliceOp>> groupedSliceOps;
  for (tensor::ExtractSliceOp sliceOp : sliceOps)
    groupedSliceOps[getSliceKey(sliceOp)].push_back(sliceOp);

  // merge groups
  // E.g. if group A consists of slices on dim 0, group B consists of slices on
  // dim 0 & 1, A will be merged to B.
  SmallVector<int64_t> keys;
  for (auto it : groupedSliceOps)
    keys.push_back(it.first);
  auto countOnes = [](int64_t num) {
    int64_t cnt = 0;
    while (num) {
      num &= (num - 1);
      cnt++;
    }
    return cnt;
  };
  std::sort(keys.begin(), keys.end(),
            [&](int64_t a, int64_t b) { return countOnes(a) > countOnes(b); });
  mlir::DenseSet<int64_t> visitedKeys;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (visitedKeys.contains(i))
      continue;
    visitedKeys.insert(i);
    for (size_t j = i + 1; j < keys.size(); ++j) {
      if (visitedKeys.contains(j))
        continue;
      if ((keys[i] & keys[j]) == keys[j]) {
        // merge group j to group i
        visitedKeys.insert(j);
        for (tensor::ExtractSliceOp jOp : groupedSliceOps[keys[j]])
          groupedSliceOps[keys[i]].push_back(jOp);
        groupedSliceOps.erase(keys[j]);
      }
    }
  }

  for (auto it : groupedSliceOps) {
    SmallVector<tensor::ExtractSliceOp> &curSliceOps = it.second;

    // No need to merge if there's only one op
    if (curSliceOps.size() == 1) {
      mergedSliceOps.push_back(curSliceOps[0]);
      continue;
    }

    int64_t rank = curSliceOps[0].getSourceType().getRank();
    Value source = curSliceOps[0].getSource();
    Location loc = source.getLoc();
    // declare lower/upper bounds for the merged slice op
    SmallVector<Value> lowerBounds, upperBounds;
    lowerBounds.resize(rank, nullptr);
    upperBounds.resize(rank, nullptr);

    for (tensor::ExtractSliceOp sliceOp : curSliceOps) {
      // all the strides are expected to be 1
      ArrayRef<int64_t> strides = sliceOp.getStaticStrides();
      assert(llvm::all_of(strides, [](int64_t x) { return x == 1; }));

      // Merge the static and dynamic values
      SmallVector<Value> curOffsets = getValueOrCreateConstantIndexOp(
          builder, loc,
          getMixedValues(sliceOp.getStaticOffsets(), sliceOp.getOffsets(),
                         builder));
      SmallVector<Value> curSizes = getValueOrCreateConstantIndexOp(
          builder, loc,
          getMixedValues(sliceOp.getStaticSizes(), sliceOp.getSizes(),
                         builder));

      // calculate the lower/upper bounds
      for (auto it : llvm::enumerate(llvm::zip(curOffsets, curSizes))) {
        int64_t idx = it.index();
        Value offset = std::get<0>(it.value());
        Value size = std::get<1>(it.value());
        Value newOffset = lowerBounds[idx]
                              ? builder.createOrFold<arith::MinSIOp>(
                                    sliceOp.getLoc(), lowerBounds[idx], offset)
                              : offset;
        lowerBounds[idx] = newOffset;
        Value upperBound =
            builder.createOrFold<arith::AddIOp>(loc, offset, size);
        Value newUpperBound;
        // Reuse old upper bound if it's the same as the new upper bound.
        // NOTE: This is just a folding rule for arith::MaxSI.
        // This is useful for case like:
        //   upperBound = arith::MaxSI(old, new)
        //   extractSlice = tensor::expand_slice(%input)[..., upperBound,
        //   ...][...] -> ?xf32
        // which result shape is dynamic.
        // if old == new, tensor::expand_slice would be static, if old can be
        // fold to a constant
        if (upperBounds[idx]) {
          if (auto oldUpperBound =
                  dyn_cast<arith::AddIOp>(upperBounds[idx].getDefiningOp())) {
            bool sameOffset = oldUpperBound.getLhs() == offset;
            auto sizeVal =
                dyn_cast_or_null<arith::ConstantIndexOp>(size.getDefiningOp());
            auto oldSizeVal = dyn_cast_or_null<arith::ConstantIndexOp>(
                oldUpperBound.getRhs().getDefiningOp());
            if (sizeVal && oldSizeVal && sameOffset &&
                sizeVal.value() == oldSizeVal.value()) {
              // newUpperBound == upperBound
              newUpperBound = upperBound;
            } else {
              newUpperBound = builder.createOrFold<arith::MaxSIOp>(
                  loc, upperBounds[idx], upperBound);
            }
          } else {
            newUpperBound = builder.createOrFold<arith::MaxSIOp>(
                loc, upperBounds[idx], upperBound);
          }
        } else {
          newUpperBound = upperBound;
        }
        upperBounds[idx] = newUpperBound;
      }
    }

    // calculate the merged op's slice sizes
    SmallVector<Value> sizes;
    sizes.reserve(rank);
    for (auto it : llvm::zip(lowerBounds, upperBounds)) {
      Value lb = std::get<0>(it);
      Value ub = std::get<1>(it);
      sizes.push_back(
          builder.createOrFold<arith::SubIOp>(source.getLoc(), ub, lb));
    }

    // create the merged slice op
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    tensor::ExtractSliceOp mergedSliceOp =
        builder.create<tensor::ExtractSliceOp>(
            source.getLoc(), source, getAsOpFoldResult(lowerBounds),
            getAsOpFoldResult(sizes), strides);

    // replace the old slice op with merged + sub slice ops
    for (tensor::ExtractSliceOp sliceOp : curSliceOps) {
      SmallVector<Value> subLowerBounds;
      subLowerBounds.reserve(rank);
      SmallVector<Value> curOffsets = getValueOrCreateConstantIndexOp(
          builder, sliceOp.getLoc(),
          getMixedValues(sliceOp.getStaticOffsets(), sliceOp.getOffsets(),
                         builder));
      SmallVector<OpFoldResult> subSizes =
          getMixedValues(sliceOp.getStaticSizes(), sliceOp.getSizes(), builder);
      for (auto it : llvm::zip(curOffsets, lowerBounds)) {
        Value curOffset = std::get<0>(it);
        Value lb = std::get<1>(it);
        Value subLb = builder.createOrFold<arith::SubIOp>(sliceOp.getLoc(),
                                                          curOffset, lb);
        subLowerBounds.push_back(subLb);
      }
      tensor::ExtractSliceOp subSliceOp =
          builder.create<tensor::ExtractSliceOp>(
              sliceOp.getLoc(), sliceOp.getResultType(),
              mergedSliceOp.getResult(), getAsOpFoldResult(subLowerBounds),
              subSizes, strides);
      sliceOp->replaceAllUsesWith(subSliceOp);
    }

    mergedSliceOps.push_back(mergedSliceOp);
  }

  return mergedSliceOps;
}

} // namespace

FailureOr<scf::SCFTileAndFuseResult>
mlir::scf::tileConsumerAndFuseProducerUsingSCFForOpExt(
    RewriterBase &rewriter, TilingInterface consumer,
    mlir::transform::TransformState &state, ArrayRef<Operation *> stopOps,
    const scf::SCFTileAndFuseOptions &options, bool simplifyLoopIter,
    bool keepIntermediate) {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        consumer, "invalid pattern for op with no results");
  }

  mlir::DenseMap<Value, int64_t> val2AllUses =
      getNumberOfUsesFromRoot(consumer.getOperation());

  DenseSet<Operation *> stopSet(stopOps.begin(), stopOps.end());
  if (stopSet.contains(consumer.getOperation()))
    return success();

  // 1. First tile the consumer.
  scf::SCFTileAndFuseResult tileAndFuseResult;
  llvm::SmallDenseMap<Value, int64_t> yieldedValueToResultNumber;
  {
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, consumer, options.tilingOptions);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");

    if (failed(isValidTiling(tilingResult->tiledOps.back()))) {
      return rewriter.notifyMatchFailure(
          consumer, "failed to tile consumer due to invalid tiling");
    }

    for (auto innerOp : tilingResult->tiledOps) {
      tileAndFuseResult.tiledAndFusedOps.insert(innerOp);
    }

    tileAndFuseResult.loops = std::move(tilingResult->loops);
    for (const auto &result : llvm::enumerate(
             llvm::zip(consumer->getResults(), tilingResult->replacements))) {
      tileAndFuseResult.replacements[std::get<0>(result.value())] =
          std::get<1>(result.value());
      yieldedValueToResultNumber[tilingResult->tiledOps.back()->getResult(
          result.index())] = result.index();
    }
  }

  // If there are no loops generated, fusion is immaterial.
  if (tileAndFuseResult.loops.empty())
    return tileAndFuseResult;

  llvm::SmallVector<std::pair<Operation *, Operation *>> fusedOps;
  fusedOps.emplace_back(consumer.getOperation(),
                        tileAndFuseResult.tiledAndFusedOps.back());

  DenseMap<Value, int64_t> val2CurrentUses;
  DenseMap<Value, SmallVector<tensor::ExtractSliceOp>> val2OriginalSliceOps;
  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of the
  //    untiled operation. Create a worklist of these `tensor.extract_slice`
  //    operations. If the producers of the source of the `tensor.extract_slice`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices =
      [&](Operation *fusedOp, std::deque<tensor::ExtractSliceOp> &candidates) {
        for (auto &opOperand : fusedOp->getOpOperands()) {
          bool validFusibleProducerOp =
              succeeded(isValidFusibleProducerOp(opOperand, fusedOp));

          if (auto sliceOp =
                  opOperand.get().getDefiningOp<tensor::ExtractSliceOp>()) {
            auto [srcResult, destinationIterArg] =
                getUntiledProducerFromSliceSource(&sliceOp->getOpOperand(0),
                                                  tileAndFuseResult.loops);
            if (!srcResult)
              continue;
            Operation *srcOp = srcResult.getOwner();
            if (!isa_and_nonnull<TilingInterface>(srcOp)) {
              continue;
            }
            assert(val2AllUses.contains(srcResult));
            val2CurrentUses[srcResult]++;
            if (validFusibleProducerOp)
              val2OriginalSliceOps[srcResult].push_back(sliceOp);
            if (validFusibleProducerOp && !stopSet.contains(srcOp) &&
                val2CurrentUses[srcResult] == val2AllUses[srcResult]) {
              SmallVector<tensor::ExtractSliceOp> mergedSliceOps =
                  mergeSliceOps(val2OriginalSliceOps[srcResult]);
              for (tensor::ExtractSliceOp sliceOp : mergedSliceOps) {
                LLVM_DEBUG(DBGS() << "enqueue cadidate for source: "
                                  << srcResult << "\n");
                candidates.push_back(sliceOp);
              }
            }
          }
        }
      };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tileAndFuseResult.tiledAndFusedOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // 2a. Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    LLVM_DEBUG(DBGS() << "deque candidate " << candidateSliceOp << "\n");
    candidates.pop_front();

    // 2b. Get the producer of the source (potentially walking through
    // `iter_args` of nested `scf.for`)
    auto [fusibleProducer, destinationIterArg] =
        getUntiledProducerFromSliceSource(&candidateSliceOp->getOpOperand(0),
                                          tileAndFuseResult.loops);
    if (!fusibleProducer) {
      LLVM_DEBUG(DBGS() << "skip since no fusibleProducer\n");
      continue;
    }

    // 2c. Generate the tiled implementation of the producer of the source
    rewriter.setInsertionPoint(candidateSliceOp);

    FailureOr<TilingResult> fusedTilingResult =
        tensor::replaceExtractSliceWithTiledProducer(rewriter, candidateSliceOp,
                                                     fusibleProducer);

    if (failed(fusedTilingResult)) {
      LLVM_DEBUG(DBGS() << "skip since no ExtractSlice of producuer for "
                        << fusibleProducer << "\n");
      continue;
    }

    Value fusedProducerValue = fusedTilingResult->tiledValues[0];
    Operation *fusedProducerOp = fusedProducerValue.getDefiningOp();
    // When tiling tensor.pad op, it will generate the IR below:
    // tensor.extract ...
    // tensor.pad ...
    // tensor.cast ...
    // So we need to find the correct `fusedProducerOp`
    if (auto castOp = dyn_cast<tensor::CastOp>(fusedProducerOp)) {
      Operation *castSrcOp = castOp.getSource().getDefiningOp();
      if (tensor::canFoldIntoProducerOp(castOp) ||
          isa<tensor::PadOp>(castSrcOp))
        fusedProducerOp = castSrcOp;
    }

    if (failed(confirmValidFusion(rewriter, fusibleProducer, fusedProducerOp,
                                  candidateSliceOp))) {
      LLVM_DEBUG(DBGS() << "skip since failing confirmValidFusion\n");
      continue;
    }

    rewriter.replaceOp(candidateSliceOp, fusedProducerValue);
    LLVM_DEBUG(DBGS() << "fusedProducerValue: " << fusedProducerValue << "\n");

    // Don't need the following steps if `fusibleProducer.getOwner()` doesn't
    // implement DestinationStyleOpInterface
    if (!isa<DestinationStyleOpInterface>(fusibleProducer.getOwner())) {
      addCandidateSlices(fusedProducerOp, candidates);
      continue;
    }

    // Always create result slices here
    // Later in step 3, we will remove redundant ones

    if (failed(createResultSlices(
            rewriter, fusibleProducer.getOwner(), fusedProducerOp,
            candidateSliceOp, tileAndFuseResult.loops,
            tileAndFuseResult.replacements, destinationIterArg))) {
      LLVM_DEBUG(DBGS() << "skip since failing createResultSlices for "
                        << fusibleProducer << "\n");
      continue;
    }

    // 2d. The operands of the fused producer might themselved be slices of
    //     values produced by operations that implement the `TilingInterface`.
    //     Add these operations to the worklist.
    // put fused one in tileAndFuseResult
    // insert tiledAndFusedOps only when it is not in it.
    if (!tileAndFuseResult.fusedProducers.contains(
            fusibleProducer.getOwner())) {
      tileAndFuseResult.fusedProducers.insert(fusibleProducer.getOwner());
      tileAndFuseResult.tiledAndFusedOps.insert(fusedProducerOp);
    }
    fusedOps.emplace_back(fusibleProducer.getOwner(), fusedProducerOp);
    addCandidateSlices(fusedProducerOp, candidates);

    // 2e. If the slice is for a destination operand, for example,
    //
    // ```mlir
    // %0 = linalg.init
    // %1 = linalg.fill .. outs(%0 : )
    // %2 = scf.for .. iter_args(%arg0 = %1) {
    //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %4 = tensor.extract_slice %arg1 [..]
    //     .. = linalg.matmul .. outs(%4 : )
    //   }
    // }
    // ```
    //
    // the IR is currently
    //
    // ```
    // %0 = linalg.init
    // %1 = linalg.fill
    // %2 = scf.for .. iter_args(%arg0 = %1 /* incorrect value */ ) {
    //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %4 = tensor.extract_slice %0 /*incorrect value */ [..]
    //     %5 = linalg.fill .. outs(%4 : )
    //     .. = linalg.matmul .. outs(%5 : )
    //   }
    // }
    // ```
    //
    // The untiled `linalg.fill` is still used as the `init_value` since it
    // was originally a destination operand of the untiled `linalg.matmul`.
    // When fusing an operand that is a destination operand.
    //   - Update the iter_arg of the outer most loop to use the destination
    //     of the untiled producer.
    //   - Update the destination of the slice of the tiled producer generated
    //     to use the same basic block argument as the slice that was used to
    //     generate inplace the tiled implementation of the producer.
    // With this the IR will be.
    //
    // ```
    // %0 = linalg.init
    // %1 = scf.for .. iter_args(%arg0 = %0 /* corrected value */ ) {
    //   %2 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %3 = tensor.extract_slice %arg1 /* corrected value */ [..]
    //     %4 = linalg.fill .. outs(%3 : )
    //     .. = linalg.matmul .. outs(%4 : )
    //   }
    // }
    // ```
    // TODO: This can be modeled better if the `DestinationStyleOpInterface`.
    // Update to use that when it does become available.
    scf::ForOp outerMostLoop = tileAndFuseResult.loops.front();
    std::optional<unsigned> iterArgNumber;
    if (destinationIterArg) {
      iterArgNumber =
          outerMostLoop.getIterArgNumberForOpOperand(**destinationIterArg);
    }
    if (iterArgNumber) {
      int64_t resultNumber = fusibleProducer.getResultNumber();
      if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(
              fusibleProducer.getOwner())) {
        outerMostLoop.setIterArg(
            *iterArgNumber, dstOp.getTiedOpOperand(fusibleProducer)->get());
      }

      if (auto dstOp =
              fusedProducerValue.getDefiningOp<DestinationStyleOpInterface>()) {
        scf::ForOp innerMostLoop = tileAndFuseResult.loops.back();
        updateDestinationOperandsForTiledOp(
            rewriter, dstOp.getDpsInitOperand(resultNumber)->get(),
            innerMostLoop.getRegionIterArgs()[*iterArgNumber]);
      }
    }

  } // while (!candidates.empty())

  // 3. topologically sort the ops since the order was corrupted in the slice
  // merging step
  for (auto &loop : tileAndFuseResult.loops) {
    if (!sortTopologically(loop.getBody()))
      return rewriter.notifyMatchFailure(consumer, "topological sort fails.");
  }

  // 4. clean up loops args and unused loop carries
  if (!keepIntermediate) {
    // collect all iterArgToOperand for quick access later
    // iterArgToOperand as mapping from Loop's RegionIterArgs to IterOperands
    llvm::DenseMap<Value, Value> iterArgToOperand;
    for (auto &forOp : tileAndFuseResult.loops) {
      for (auto it : llvm::zip(forOp.getRegionIterArgs(), // iter inside region
                               forOp.getIterOperands()    // iter from outside
                               )) {
        iterArgToOperand.try_emplace(std::get<0>(it), std::get<1>(it));
      }
    }

    assert(tileAndFuseResult.loops.size() > 0);
    // check getLoopIteratorTypes for each fusedOp
    // if parallel, corresponding getRegionIterArgs will be simplified
    unsigned resultOffset = 0;

    llvm::DenseSet<Operation *> unfusedOpsSet;
    for (auto &p : fusedOps) {
      Operation *unfusedOp = p.first;
      unfusedOpsSet.insert(unfusedOp);
    }

    for (const auto &p : fusedOps) {
      auto unfusedOp = p.first;
      auto fusedOp = p.second;
      auto numResult = fusedOp->getNumResults();

      // analyze LoopIteratorTypes before using
      auto loopIterTypes =
          getLoopIteratorTypes(fusedOp, tileAndFuseResult.loops);
      if (failed(loopIterTypes)) {
        LLVM_DEBUG(DBGS() << "skip clean-up due to no loopIterTypes for "
                          << fusedOp << "\n");
        resultOffset += numResult;
        continue;
      }

      for (unsigned i = 0; i < unfusedOp->getNumResults(); ++i) {
        auto result = unfusedOp->getResult(i);

        auto effectiveUseCnt =
            llvm::count_if(result.getUses(), [&](OpOperand &opOperand) {
              if (unfusedOpsSet.contains(opOperand.getOwner()))
                return false;

              if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(
                      opOperand.getOwner())) {
                return !dstOp.isDpsInit(&opOperand);
              }

              return !isa<tensor::DimOp>(opOperand.getOwner());
            });

        bool hasZeroOutsideUse = effectiveUseCnt == 0;

        auto confirmAllParallel = [&](size_t loopCnt) {
          bool allParallel = true;
          for (size_t idx = 0; idx <= loopCnt; ++idx) {
            auto &maybeIterTy = (*loopIterTypes)[idx];
            if (allParallel &&
                !(maybeIterTy.has_value() &&
                  *maybeIterTy == utils::IteratorType::parallel)) {
              allParallel = false;
            }
          }
          return allParallel;
        };

        for (int64_t loopIdx = tileAndFuseResult.loops.size() - 1; loopIdx >= 0;
             loopIdx -= 1) {

          // update collection every iteration, since it might be replaced.
          SmallPtrSet<Operation *, 8> opCollection;
          SmallPtrSet<Value, 16> valCollection;
          opCollection.insert(fusedOp);
          // get all producer and consumer slices' op and value
          getProducerAndConsumerTensorSlices(fusedOp, iterArgToOperand,
                                             opCollection, valCollection);

          auto &forOp = tileAndFuseResult.loops[loopIdx];
          bool confirmedAllParallel = confirmAllParallel(loopIdx);

          auto iterArg = forOp.getRegionIterArg(resultOffset + i);
          auto iterOperand = forOp.getIterOperands()[resultOffset + i];

          if (isResultLoopInvariant(unfusedOp, i, hasZeroOutsideUse,
                                    confirmedAllParallel)) {
            iterArg.replaceUsesWithIf(iterOperand, [&](OpOperand &use) {
              return (opCollection.contains(use.getOwner()) ||
                      valCollection.contains(use.get()));
            });
          }

          // The following replace is used to optimize the following IR:
          //
          // %0 = tensor.empty
          // scf.for ... (%arg0 = %0, ...)
          //    %1 = tensor.extract %arg0
          //    "use"(%1)...
          //
          // to
          //
          // scf.for ...
          //   %0 = tensor.empty
          //   "use"(%0)
          if (simplifyLoopIter &&
              isResultLoopInvariant(unfusedOp, i, true, confirmedAllParallel)) {
            iterArg.replaceUsesWithIf(iterOperand, [&](OpOperand &use) {
              return isa<tensor::ExtractSliceOp>(use.getOwner());
            });
          }
        } // int64_t loopIdx > 0
      }   // for i < unfusedOp->getNumResults()
      resultOffset += numResult;
    } // for (const auto &p : fusedOps)
  }

  return tileAndFuseResult;
}

namespace {

static LogicalResult checkRootsAndTileOptions(ArrayRef<Value> tensors,
                                              const TilingOptions &options) {
  if (tensors.size() == 0) {
    LLVM_DEBUG(DBGS() << "Expect at least one root tensor.\n");
    return failure();
  }
  for (Value tensor : tensors) {
    TilingInterface rootOp = tensor.getDefiningOp<TilingInterface>();
    if (!rootOp) {
      LLVM_DEBUG(DBGS() << "invalid pattern for tensor with no defining op of "
                           "tiling interface");
      return failure();
    }
  }
  if (!options.isValid()) {
    LLVM_DEBUG(DBGS() << "tile options are not valid\n");
    return failure();
  }

  if (options.isTileSizes() && tensors.size() != 1)
    return failure();

  return success();
}

} // namespace

FailureOr<scf::SCFTileAndFuseResult>
mlir::scf::tileConsumerArrayAndFuseProducerGreedilyUsingSCFFor(
    RewriterBase &rewriter, ArrayRef<Value> tensors,
    const TilingOptions &options, TileFuncType tileFunc,
    bool expectWholeGraphFusion) {
  if (failed(checkRootsAndTileOptions(tensors, options))) {
    LLVM_DEBUG(DBGS() << "check root and tile options failed\n");
    return failure();
  }

  // All ops will be topologically sorted at the end
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(tensors[0].getDefiningOp());
  Location loc = tensors[0].getLoc();

  SmallVector<OpFoldResult> tileNums;
  if (options.isTileNums())
    tileNums = options.tileNums;
  else
    assert(0 && "currently only support tile nums option");
  const SmallVector<int64_t> &interchange = options.interchange;
  scf::SCFTileAndFuseResult tileAndFuseResult;
  llvm::DenseSet<Value> rootTensorsSet{tensors.begin(), tensors.end()};
  mlir::DenseMap<Value, int64_t> val2Uses = getNumberOfUsesFromRoots(tensors);

  // create scf.for ops
  SmallVector<OpFoldResult> validTileNums = getValidTileNums(tileNums);
  tileAndFuseResult.loops = scf::createNestedEmptyScfForOpsWithZeroLbAndOneStep(
      rewriter, loc, validTileNums);

  // If there are no loops generated, fusion is immaterial.
  if (tileAndFuseResult.loops.empty())
    return tileAndFuseResult;

  // 1. First tile all the root tensors with no uses
  if (tileFunc == nullptr) {
    // Use the default tile func
    tileFunc = tileToExistedLoops;
  }
  for (Value tensor : tensors) {
    if (val2Uses[tensor] == 0) {
      TilingInterface tileableOp = tensor.getDefiningOp<TilingInterface>();
      if (failed(tileFunc(rewriter, tileableOp, tileNums, interchange,
                          options.useDistributedStyle, tileAndFuseResult)))
        return rewriter.notifyMatchFailure(tileableOp, "failed to tile");
    }
  }

  LLVM_DEBUG({
    if (!tileAndFuseResult.loops.empty()) {
      tileAndFuseResult.loops.front().dump();
      llvm::dbgs() << "\n";
    }
  });

  DenseMap<Value, SmallVector<tensor::ExtractSliceOp>> val2OriginalSliceOps;
  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of the
  //    untiled operation. Create a worklist of these `tensor.extract_slice`
  //    operations. If the producers of the source of the `tensor.extract_slice`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices =
      [&](Operation *fusedOp, std::deque<tensor::ExtractSliceOp> &candidates) {
        for (auto &opOperand : fusedOp->getOpOperands()) {
          bool validFusibleProducerOp =
              succeeded(isValidFusibleProducerOp(opOperand, fusedOp));

          if (auto sliceOp =
                  opOperand.get().getDefiningOp<tensor::ExtractSliceOp>()) {
            OpResult srcResult = std::get<0>(getUntiledProducerFromSliceSource(
                &sliceOp->getOpOperand(0), tileAndFuseResult.loops));
            if (!srcResult)
              continue;
            Operation *srcOp = srcResult.getOwner();
            if (!isa_and_nonnull<TilingInterface>(srcOp)) {
              continue;
            }
            auto it = val2Uses.find(srcResult);
            assert(it != val2Uses.end() && it->second > 0);
            val2Uses[srcResult]--;
            if (validFusibleProducerOp)
              val2OriginalSliceOps[srcResult].push_back(sliceOp);
            if (val2Uses[srcResult] == 0) {
              SmallVector<tensor::ExtractSliceOp> mergedSliceOps =
                  mergeSliceOps(val2OriginalSliceOps[srcResult]);
              for (tensor::ExtractSliceOp sliceOp : mergedSliceOps) {
                LLVM_DEBUG(DBGS() << "enqueue cadidate for source: "
                                  << srcResult << "\n");
                candidates.push_back(sliceOp);
              }
            }
          }
        }
      };

  std::deque<tensor::ExtractSliceOp> candidates;
  for (Operation *fusedOp : tileAndFuseResult.tiledAndFusedOps)
    addCandidateSlices(fusedOp, candidates);

  while (!candidates.empty()) {
    // 3a. Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    Value tileableTensor = candidateSliceOp.getSource();
    candidates.pop_front();

    // 3b. Get the producer of the source (potentially walking through
    // `iter_args` of nested `scf.for`)
    auto [fusibleProducer, destinationIterArg] =
        getUntiledProducerFromSliceSource(&candidateSliceOp->getOpOperand(0),
                                          tileAndFuseResult.loops);
    if (!fusibleProducer) {
      LLVM_DEBUG(DBGS() << "skip since no fusibleProducer\n");
      continue;
    }

    // 3c. Generate the tiled implementation of the producer of the source
    rewriter.setInsertionPoint(candidateSliceOp);
    FailureOr<TilingResult> fusedTilingResult =
        tensor::replaceExtractSliceWithTiledProducer(rewriter, candidateSliceOp,
                                                     fusibleProducer);
    if (failed(fusedTilingResult)) {
      LLVM_DEBUG(DBGS() << "skip since no ExtractSlice of producuer for "
                        << fusibleProducer << "\n");
      continue;
    }

    Value fusedProducerValue = fusedTilingResult->tiledValues[0];
    Operation *fusedProducerOp = fusedProducerValue.getDefiningOp();
    // When tiling tensor.pad op, it will generate the IR below:
    // tensor.extract ...
    // tensor.pad ...
    // tensor.cast ...
    // So we need to find the correct `fusedProducerOp`
    if (auto castOp = dyn_cast<tensor::CastOp>(fusedProducerOp)) {
      Operation *castSrcOp = castOp.getSource().getDefiningOp();
      if (tensor::canFoldIntoProducerOp(castOp) ||
          isa<tensor::PadOp>(castSrcOp))
        fusedProducerOp = castSrcOp;
    }

    if (failed(confirmValidFusion(rewriter, fusibleProducer, fusedProducerOp,
                                  candidateSliceOp))) {
      LLVM_DEBUG(DBGS() << "skip since failing confirmValidFusion\n");
      continue;
    }

    rewriter.replaceOp(candidateSliceOp, fusedProducerValue);
    LLVM_DEBUG(DBGS() << "fusedProducerValue: " << fusedProducerValue << "\n");

    // Don't need the following steps if `fusibleProducer.getOwner()` doesn't
    // implement DestinationStyleOpInterface
    if (!isa<DestinationStyleOpInterface>(fusibleProducer.getOwner())) {
      addCandidateSlices(fusedProducerOp, candidates);
      continue;
    }

    if (rootTensorsSet.contains(tileableTensor)) {
      if (failed(createResultSlices(
              rewriter, fusibleProducer.getOwner(), fusedProducerOp,
              candidateSliceOp, tileAndFuseResult.loops,
              tileAndFuseResult.replacements, destinationIterArg))) {
        LLVM_DEBUG(DBGS() << "skip since failing createResultSlices for "
                          << fusibleProducer << "\n");
        continue;
      }
    }

    // 2d. The operands of the fused producer might themselved be slices of
    //     values produced by operations that implement the `TilingInterface`.
    //     Add these operations to the worklist.
    // put fused one in tileAndFuseResult
    // insert tiledAndFusedOps only when it is not in it.
    if (!tileAndFuseResult.fusedProducers.contains(
            fusibleProducer.getOwner())) {
      tileAndFuseResult.fusedProducers.insert(fusibleProducer.getOwner());
      tileAndFuseResult.tiledAndFusedOps.insert(fusedProducerOp);
    }
    addCandidateSlices(fusedProducerOp, candidates);

    // 2e. If the slice is for a destination operand, for example,
    //
    // ```mlir
    // %0 = linalg.init
    // %1 = linalg.fill .. outs(%0 : )
    // %2 = scf.for .. iter_args(%arg0 = %1) {
    //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %4 = tensor.extract_slice %arg1 [..]
    //     .. = linalg.matmul .. outs(%4 : )
    //   }
    // }
    // ```
    //
    // the IR is currently
    //
    // ```
    // %0 = linalg.init
    // %1 = linalg.fill
    // %2 = scf.for .. iter_args(%arg0 = %1 /* incorrect value */ ) {
    //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %4 = tensor.extract_slice %0 /*incorrect value */ [..]
    //     %5 = linalg.fill .. outs(%4 : )
    //     .. = linalg.matmul .. outs(%5 : )
    //   }
    // }
    // ```
    //
    // The untiled `linalg.fill` is still used as the `init_value` since it
    // was originally a destination operand of the untiled `linalg.matmul`.
    // When fusing an operand that is a destination operand.
    //   - Update the iter_arg of the outer most loop to use the destination
    //     of the untiled producer.
    //   - Update the destination of the slice of the tiled producer generated
    //     to use the same basic block argument as the slice that was used to
    //     generate inplace the tiled implementation of the producer.
    // With this the IR will be.
    //
    // ```
    // %0 = linalg.init
    // %1 = scf.for .. iter_args(%arg0 = %0 /* corrected value */ ) {
    //   %2 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %3 = tensor.extract_slice %arg1 /* corrected value */ [..]
    //     %4 = linalg.fill .. outs(%3 : )
    //     .. = linalg.matmul .. outs(%4 : )
    //   }
    // }
    // ```
    // TODO: This can be modeled better if the `DestinationStyleOpInterface`.
    // Update to use that when it does become available.
    scf::ForOp outerMostLoop = tileAndFuseResult.loops.front();
    std::optional<unsigned> iterArgNumber;
    if (destinationIterArg) {
      iterArgNumber =
          outerMostLoop.getIterArgNumberForOpOperand(**destinationIterArg);
    }
    if (iterArgNumber) {
      int64_t resultNumber = fusibleProducer.getResultNumber();
      if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(
              fusibleProducer.getOwner())) {
        outerMostLoop.setIterArg(
            *iterArgNumber, dstOp.getTiedOpOperand(fusibleProducer)->get());
      }

      if (auto dstOp =
              fusedProducerValue.getDefiningOp<DestinationStyleOpInterface>()) {
        scf::ForOp innerMostLoop = tileAndFuseResult.loops.back();
        updateDestinationOperandsForTiledOp(
            rewriter, dstOp.getDpsInitOperand(resultNumber)->get(),
            innerMostLoop.getRegionIterArgs()[*iterArgNumber]);
      }
    }
  } // while (!candidates.empty())

  // 3. topologically sort the ops since the order was corrupted in the slice
  // merging step
  for (auto &loop : tileAndFuseResult.loops) {
    if (!sortTopologically(loop.getBody()))
      return rewriter.notifyMatchFailure(loc, "topological sort fails.");
  }

  // 4. if expectWholeGraphFusion is true
  if (expectWholeGraphFusion) {
    for (Operation *op : tileAndFuseResult.tiledAndFusedOps) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        tensor::ExtractSliceOp sliceOp =
            llvm::dyn_cast<tensor::ExtractSliceOp>(defOp);
        if (!sliceOp)
          continue;
        Value sliceInput = sliceOp.getSource();
        Operation *sliceInputDefOp = sliceInput.getDefiningOp();
        if (!sliceInputDefOp)
          continue;
        if (!llvm::isa<tensor::EmptyOp, linalg::FillOp>(sliceInputDefOp))
          return rewriter.notifyMatchFailure(sliceInputDefOp,
                                             "expect whole graph fusion");
      }
    }
  }

  return tileAndFuseResult;
}

namespace mlir {
namespace linalg_ext {
// Marker used as attribute name in generated Linalg rewriting
// transformations.
const StringLiteral LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

LinalgTransformationFilter::LinalgTransformationFilter(
    const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {
  if (f)
    filters.push_back(f);
}

LogicalResult
LinalgTransformationFilter::checkAndNotify(PatternRewriter &rewriter,
                                           Operation *op) const {
  if (llvm::any_of(filters,
                   [&](const FilterFunction &f) { return failed(f(op)); }))
    return failure();

  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty() || matchByDefault)
      return success();

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction)
    if (attr.getValue() == filter)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void LinalgTransformationFilter::replaceLinalgTransformationFilter(
    PatternRewriter &rewriter, Operation *op) const {
  if (replacement.has_value())
    op->setAttr(LinalgTransforms::kLinalgTransformMarker, *replacement);
  else
    op->removeAttr(
        rewriter.getStringAttr(LinalgTransforms::kLinalgTransformMarker));
}

bool LinalgTransformationFilter::hasReplacementFilter(Operation *op) const {
  if (!replacement)
    return false;
  auto attr = op->getAttr(LinalgTransforms::kLinalgTransformMarker)
                  .dyn_cast<StringAttr>();
  return attr && attr == *replacement;
}
} // namespace linalg_ext
} // namespace mlir
