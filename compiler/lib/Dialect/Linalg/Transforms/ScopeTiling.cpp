//===- ScopeTiling.cpp ----------------------------------------*--- C++ -*-===//
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
// Some code comes from LinalgTiling.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/Tiling.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/TilingUtils.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/MemUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-scope-tiling"

namespace {

#if 0

static bool isReduction(mlir::linalg::LinalgOp linalgOp) {
  SmallVector<StringAttr> iterTypes =
      llvm::to_vector<4>(linalgOp.iterator_types().getAsRange<StringAttr>());
  unsigned axis =
      linalgOp->getAttrOfType<IntegerAttr>(getScopeTilingAxisAttrName())
          .getInt();
  return isReductionIterator(iterTypes[axis]);
}

#if 0
void tileScopeImpl(OpBuilder &b, TileScope &ts, int64_t tileSize,
                   bool parallelizeReduction, LinalgTilingLoopType loopType) {
  // early termination
  if (ts.ops.size() == 0)
    return;





  // 1. Build the tiled loop ranges.
  //    Use lastop to create loops variables
  auto lastOp = ts.ops.back();
  unsigned lastAxis =
      lastOp->getAttrOfType<IntegerAttr>(getScopeTilingAxisAttrName()).getInt();
  unsigned lastRank =
      lastOp->getAttrOfType<IntegerAttr>(getScopeTilingRankAttrName()).getInt();
  auto loc = lastOp->getLoc();

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(lastOp);

  auto lastAllShapeSizes = lastOp.createFlatListOfOperandDims(b, loc);

  AffineMap lastShapeSizesToLoopsMap = lastOp.getShapesToLoopsMap();
  if (!lastShapeSizesToLoopsMap)
    return;

  SmallVector<Range, 4> loopRanges = makeTiledLoopRange(
      b, loc, lastShapeSizesToLoopsMap, lastAllShapeSizes, lastAxis, tileSize);

  // iteratorTypes Attribute
  SmallVector<Attribute, 4> iteratorTypes;
  iteratorTypes.push_back(lastOp.iterator_types().getValue()[lastAxis]);

  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
      AffineMap::getMultiDimIdentityMap(lastRank, b.getContext());
  // TODO support loop interchange later

  bool isParallel = true;

  for (auto &linalgOp : ts.ops) {
    if (isReduction(linalgOp)) {
      isParallel = false;
      break;
    }
  }

  // 2. Create the tiled loops.
  SmallVector<Value, 8> lbsStorage, ubsStorage, stepsStorage;
  unpackRanges(loopRanges, lbsStorage, ubsStorage, stepsStorage);
  ValueRange lbs(lbsStorage), ubs(ubsStorage), steps(stepsStorage);
  SmallVector<Value, 8> ivs(lbs.size());

  auto tiledLoopBodyBuilder = [&](OpBuilder &b, Location loc,
                                  ValueRange loopIvs) {
    ivs.assign(loopIvs.begin(), loopIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<Value, 4> interchangedIvs;

    // TODO support loop interchange later
    interchangedIvs.assign(ivs.begin(), ivs.end());

    // go through all ops
    for (auto &linalgOp : ts.ops) {
      unsigned axis =
          linalgOp->getAttrOfType<IntegerAttr>(getScopeTilingAxisAttrName())
              .getInt();
      unsigned rank =
          linalgOp->getAttrOfType<IntegerAttr>(getScopeTilingRankAttrName())
              .getInt();

      auto localAllShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
      AffineMap localShapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();

      auto localSizeBounds = applyMapToValues(b, loc, localShapeSizesToLoopsMap,
                                              localAllShapeSizes);

      // get all values for now
      // TODO: relax this later
      SmallVector<Value> valuesToTile = linalgOp.getInputAndOutputOperands();

      SmallVector<Value, 4> tileSizes =
          createTileSize(b, loc, axis, rank, tileSize);

      SmallVector<Value, 4> localTiledOperands = makeTiledShapes(
          b, loc, linalgOp, valuesToTile, interchangedIvs, tileSizes,
          localSizeBounds, /*omitPartialTileCheck=*/false);

      SmallVector<Type, 4> resultTensorTypes;
      for (OpOperand *opOperand : linalgOp.getOutputTensorOperands()) {
        resultTensorTypes.push_back(
            localTiledOperands[opOperand->getOperandNumber()].getType());
      }

      // if enabling parallelize reduction and the loop is reduction
      // we need to alloc a new buffer and perform all reduce
      if (parallelizeReduction && isReduction(linalgOp)) {
        SmallVector<Value> intermediates;
        SmallVector<Value> outputs;
        for (size_t i = linalgOp.getInputBufferOperands().size();
             i < localTiledOperands.size(); ++i) {
          outputs.push_back(localTiledOperands[i]); // record all one
          auto maybeValue = createAlloc(b, localTiledOperands[i]);
          intermediates.push_back(maybeValue.getValue());
          localTiledOperands[i] = maybeValue.getValue();
        }

        auto cloned =
            linalgOp.clone(b, loc, resultTensorTypes, localTiledOperands);

        // remove attr
        cloned->removeAttr(getScopeTilingAxisAttrName());
        cloned->removeAttr(getScopeTilingRankAttrName());

        // support arith::AtomicRMWKind::addf for now
        // FIXME: extend it
        auto maybeAlloc = createAtomicLinalgGeneric(
            b, loc, arith::AtomicRMWKind::addf, intermediates, outputs);

        if (!maybeAlloc.hasValue())
          return;
      } else {
        auto cloned =
            linalgOp.clone(b, loc, resultTensorTypes, localTiledOperands);

        // remove attr
        cloned->removeAttr(getScopeTilingAxisAttrName());
        cloned->removeAttr(getScopeTilingRankAttrName());
      }
    }
  };

  if (parallelizeReduction)
    isParallel = true;

  if (loopType == LinalgTilingLoopType::Loops) {
    if (failed(buildSCFLoop(b, loc, isParallel, lbs.take_front(),
                            ubs.take_front(), steps.take_front(),
                            tiledLoopBodyBuilder))) {
      return;
    }
  } else if (loopType == LinalgTilingLoopType::AffineLoops) {
    // affine not support parallelizeReduction for now
    if (parallelizeReduction)
      return;

    if (failed(buildAffineLoop(b, loc, lbs.take_front(), ubs.take_front(),
                               steps.take_front(), tiledLoopBodyBuilder))) {
      return;
    }
  } else {
    // TODO support Linalg::TiledLoop later
    return;
  }

  for (auto &op : ts.ops) {
    op->erase();
  }
}
#endif

// TODO move util to other file
/**
 * find iteration index through dim and inversePermutation
 * E.g. if affineMap = (d0, d1, d2)-> (d0, d2), dim = 1
 * Then invMap = (d0, d1)->(d0, 0, d1)
 *      oneHot = (0, 1)
 *      invComposed = (0, 0, 1)
 *      iterAxis = 2
 **/
static std::optional<unsigned> getIterAxisFromDim(AffineMap affineMap,
                                             unsigned dimIndex) {
  AffineMap invMap = inverseAndBroadcastProjectedPermutation(affineMap);
  if (invMap.isEmpty())
    return std::nullopt;
  auto invComposed =
      invMap.compose(createOneHot(invMap.getNumInputs(), dimIndex));
  auto iterAxes = getAllIndicesForNonZeros(invComposed);
  // no support all-to-1 or non mapping
  if (iterAxes.size() != 1) {
    return std::nullopt;
  }
  return iterAxes[0];
}

/**
 * find dim through iteration index and permutation
 * E.g. if affineMap = (d0, d1, d2)-> (d0, d2), iterAxis = 2
 *      oneHot = (0, 0, 1)
 *      composed = (0, 1)
 *      dim = 1
 **/
static std::optional<unsigned> getDimFromIterAxis(AffineMap affineMap,
                                             unsigned iterAxis) {
  auto composed =
      affineMap.compose(createOneHot(affineMap.getNumInputs(), iterAxis));
  auto dims = getAllIndicesForNonZeros(composed);
  // no support all-to-1 or non mapping
  if (dims.size() != 1) {
    return std::nullopt;
  }
  return dims[0];
}

static bool breakScope(Operation &op) {
  return isa<LoopLikeOpInterface, scf::IfOp, AffineIfOp>(op);
}

static SmallVector<AffineMap> getIndexingMapsArray(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgOp.getIndexingMapsArray();
  } else if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return linalgExtOp.getIndexingMapsArray();
  }
  return {};
}

static unsigned getNumInputsAndOutputs(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgOp.getNumInputsAndOutputs();
  } else if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return linalgExtOp.getNumInputsAndOutputs();
  }
  return 0;
}

static unsigned getNumInputs(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgOp.getNumInputs();
  } else if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return linalgExtOp.getNumInputs();
  }
  return 0;
}

constexpr int64_t getNoDim() { return -1; }
constexpr int64_t getNoIterAxis() { return -1; }

static void invalidateDim(Operation *op,
                          llvm::DenseMap<Value, int64_t> &valueToDim) {
  auto numInputs = getNumInputs(op);
  auto numInputsAndOutputs = getNumInputsAndOutputs(op);
  for (unsigned i = numInputs; i < numInputsAndOutputs; ++i) {
    auto val = op->getOperand(i);
    valueToDim[val] = getNoDim();
  }

  auto numResults = op->getNumResults();
  for (unsigned i = 0; i < numResults; ++i) {
    auto val = op->getResult(i);
    valueToDim[val] = getNoDim();
  }
}

static LogicalResult
updateDim(Operation *op, unsigned iterAxis,
          llvm::DenseMap<Operation *, int64_t> &opToIterAxis,
          llvm::DenseMap<Value, int64_t> &valueToDim) {

  if (opToIterAxis.count(op) == 0) {
    opToIterAxis.try_emplace(op, iterAxis);
  } else if (opToIterAxis[op] != iterAxis) {
    // if iterAxis contradict return failture
    return failure();
  }

  auto indexingMapsArray = getIndexingMapsArray(op);
  auto numInputsAndOutputs = getNumInputsAndOutputs(op);
  llvm::DenseMap<Value, int64_t> localValueToDim;
  for (unsigned i = 0; i < numInputsAndOutputs; ++i) {
    auto maybeDim = getDimFromIterAxis(indexingMapsArray[i], iterAxis);
    auto val = op->getOperand(i);
    int64_t dim = maybeDim.hasValue() ? maybeDim.getValue() : getNoDim();

    if (valueToDim.count(val) == 0) {
      localValueToDim.try_emplace(val, dim);
    } else if (valueToDim[val] != dim || (localValueToDim.count(val) > 0 &&
                                          localValueToDim[val] != dim)) {
      // error
      // if a dim contradict, return failure
      invalidateDim(op, valueToDim);
      return failure();
    }
  }

  auto numResults = op->getNumResults();
  auto numInputs = getNumInputs(op);
  for (unsigned i = 0; i < numResults; ++i) {
    auto maybeDim =
        getDimFromIterAxis(indexingMapsArray[numInputs + i], iterAxis);
    auto val = op->getResult(i);
    int64_t dim = maybeDim.hasValue() ? maybeDim.getValue() : getNoDim();

    if (valueToDim.count(val) == 0) {
      localValueToDim.try_emplace(val, dim);
    } else if (valueToDim[val] != dim || (localValueToDim.count(val) > 0 &&
                                          localValueToDim[val] != dim)) {
      // if a dim contradict, return failure
      invalidateDim(op, valueToDim);
      return failure();
    }
  }

  // final merge and return success if no contradiction
  for (auto &it : localValueToDim) {
    if (valueToDim.count(it.first) == 0) {
      valueToDim.try_emplace(it.first, it.second);
    }
  }

  return success();
}

static LogicalResult
updateDim(Operation *op, llvm::DenseMap<Operation *, int64_t> &opToIterAxis,
          llvm::DenseMap<Value, int64_t> &valueToDim) {

  auto indexingMapsArray = getIndexingMapsArray(op);
  auto numInputsAndOutputs = getNumInputsAndOutputs(op);
  for (unsigned i = 0; i < numInputsAndOutputs; ++i) {
    auto val = op->getOperand(i);
    if (valueToDim.count(val) > 0) {
      auto dim = valueToDim[val];
      if (dim == getNoDim())
        continue;
      auto maybeIterAxis = getIterAxisFromDim(indexingMapsArray[i], dim);
      if (maybeIterAxis.hasValue() &&
          maybeIterAxis.getValue() != getNoIterAxis()) {
        if (failed(updateDim(op, maybeIterAxis.getValue(), opToIterAxis,
                             valueToDim))) {
          return failure();
        }
      }
    }
  }

  // succeed if
  // 1) having an axis and using the axis to update some dim
  // 2) having no axis
  return success();
}

static void collectTilingScope(func::FuncOp func, unsigned iterAxis,
                               SmallVectorImpl<TileScope> &scopes,
                               bool keepTag) {

  // only one anchor per block is supported
  SmallSet<Block *, 4> visitedBlocks;

  func.walk([&](Operation *op) {
    Block *block = op->getBlock();
    // skip non-targeting or visited block
    if (!op->hasAttr(getScopeTilingAnchorAttrName()) ||
        visitedBlocks.contains(block)) {
      return;
    }

    // TODO: maybe check legal here
    scopes.emplace_back(op);
    visitedBlocks.insert(block);

    // remove anchor after collect
    if (!keepTag) {
      op->removeAttr(getScopeTilingAnchorAttrName());
    }

    // extend ops from anchor to next branch
    bool beforeAnchor = true;
    for (auto &scopeOp : block->without_terminator()) {
      if (beforeAnchor) {
        if (op == &scopeOp) {
          beforeAnchor = false;
        }
        continue;
      }

      if (breakScope(scopeOp)) {
        break;
      }
      scopes.back().ops.push_back(&scopeOp);
    }
  });

  // update TileScope given a iterAxis
  for (auto &ts : scopes) {
    llvm::DenseMap<Value, int64_t> valueToDim;
    llvm::DenseMap<Operation *, int64_t> opToIterAxis;
    if (failed(updateDim(ts.anchorOp, iterAxis, opToIterAxis, valueToDim))) {
      LLVM_DEBUG(llvm::dbgs() << "Expect AnchorOp " << *ts.anchorOp
                              << " has iterAxis " << iterAxis << "\n");
      return;
    }

    bool changed = true;
    while (changed) {
      size_t sizeOpIterAxis = opToIterAxis.size();
      size_t sizeValueToDim = valueToDim.size();
      for (auto scopeOp : ts.ops) {

        if (opToIterAxis.count(scopeOp) > 0 &&
            opToIterAxis[scopeOp] == getNoIterAxis()) {
          continue;
        }

        if (failed(updateDim(scopeOp, opToIterAxis, valueToDim))) {
          opToIterAxis[scopeOp] = getNoIterAxis();
        }
      } // for scopeOp : ts.ops

      // check converge or not
      if (sizeOpIterAxis == opToIterAxis.size() &&
          sizeValueToDim == valueToDim.size()) {
        changed = false;
      }
    }

    // collect all valid ops
    SmallVector<Operation *> ops;
    ops.push_back(ts.anchorOp);
    for (auto scopeOp : ts.ops) {
      if (opToIterAxis.count(scopeOp) > 0 &&
          opToIterAxis[scopeOp] != getNoDim()) {
        ops.push_back(scopeOp);
      }
    }
    ts.ops = ops;
  }
}

bool isHoistUpOp(Operation *op) {
  return isa<memref::AllocOp, memref::CollapseShapeOp, memref::DimOp,
             memref::ExpandShapeOp, memref::ReshapeOp>(op);
}

bool isHoistDownOp(Operation *op) { return isa<memref::DeallocOp>(op); }

#endif

struct LinalgScopeTilingPass
    : public LinalgScopeTilingBase<LinalgScopeTilingPass> {
  LinalgScopeTilingPass() = default;
  LinalgScopeTilingPass(int64_t tileAxis, int64_t tileSize,
                        bool parallelizeReduction,
                        linalg::LinalgTilingLoopType loopType, bool keepTag) {

    this->tileAxis = tileAxis;
    this->tileSize = tileSize;
    this->parallelizeReduction = parallelizeReduction;
    this->loopType = "";
    loopTypeEnum = loopType;
    this->keepTag = keepTag;
  }

  void runOnOperation() override {
#if 0
    // early terminate when tileSize == 0
    if (tileSize == 0)
      return;

    // FIXME: comment out now, 
    //        bring back when we want to support different types
#if 0
    // parse
    LinalgTilingLoopType type =
        llvm::StringSwitch<LinalgTilingLoopType>(loopType)
            .Case("scf", LinalgTilingLoopType::Loops)
            .Case("affine", LinalgTilingLoopType::AffineLoops)
            .Case("parallel", LinalgTilingLoopType::ParallelLoops)
            .Case("tiled_loop", LinalgTilingLoopType::TiledLoops)
            .Default(loopTypeEnum);
#endif

    func::FuncOp funcOp = getOperation();

    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &postDomInfo = getAnalysis<PostDominanceInfo>();

    // hoisting
    for (auto &block : funcOp.getBody()) {
      hoistUpOpsInBlock(&block, domInfo, isHoistUpOp);
      hoistDownOpsInBlock(&block, postDomInfo, isHoistDownOp);
    }

    SmallVector<TileScope> collection;
    collectTilingScope(funcOp, tileAxis, collection, keepTag);

    // debug
    LLVM_DEBUG(llvm::dbgs() << "Debug TileScope\n");
    for (auto &ts : collection) {
      LLVM_DEBUG(llvm::dbgs() << "AnchorOp " << *ts.anchorOp << "\n");
      LLVM_DEBUG(llvm::dbgs() << "ops\n");
      for (auto op : ts.ops) {
        LLVM_DEBUG(llvm::dbgs() << *op << "\n");
      }
    }

#if 0
    OpBuilder b(funcOp.getContext());
    for (auto &ts : collection) {
      tileScopeImpl(b, ts, tileSize, parallelizeReduction, type);
    }
#endif
#endif
  }

  linalg::LinalgTilingLoopType loopTypeEnum;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgScopeTilingPass(
    int64_t tileAxis, int64_t tileSize, bool parallelizeReduction,
    linalg::LinalgTilingLoopType loopType, bool keepTag) {
  return std::make_unique<LinalgScopeTilingPass>(
      tileAxis, tileSize, parallelizeReduction, loopType, keepTag);
}
