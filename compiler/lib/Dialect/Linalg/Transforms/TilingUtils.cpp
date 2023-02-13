//===- TilingUtils.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Linalg/Transforms/TilingUtils.h"
#include "byteir/Utils/MemUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

#define K_INITIAL -999

/// Given a list of subview ranges, extract individual values for lower, upper
/// bounds and steps and put them into the corresponding vectors.
void mlir::unpackRanges(OpBuilder &builder, Location loc,
                        ArrayRef<Range> ranges, SmallVectorImpl<Value> &lbs,
                        SmallVectorImpl<Value> &ubs,
                        SmallVectorImpl<Value> &steps) {

  for (Range range : ranges) {
    lbs.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.offset));
    ubs.emplace_back(getValueOrCreateConstantIndexOp(builder, loc, range.size));
    steps.emplace_back(
        getValueOrCreateConstantIndexOp(builder, loc, range.stride));
  }
}

LogicalResult mlir::buildSCFLoop(
    OpBuilder &b, Location loc, bool isParallel, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {

  if (isParallel) {
    b.create<scf::ParallelOp>(loc, lbs.take_front(), ubs.take_front(),
                              steps.take_front(), bodyBuilder);
  } else {
    buildLoopNest(b, loc, lbs.take_front(), ubs.take_front(),
                  steps.take_front(), bodyBuilder);
  }

  return success();
}

LogicalResult mlir::buildAffineLoop(
    OpBuilder &b, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {

  llvm::SmallVector<int64_t, 4> stepLiterals;
  for (auto step : steps.take_front()) {
    auto lit = getLiteralFromConstantLike(step, K_INITIAL);
    if (lit == K_INITIAL) {
      return failure();
    }
    stepLiterals.push_back(lit);
  }

  buildAffineLoopNest(b, loc, lbs.take_front(), ubs.take_front(), stepLiterals,
                      bodyBuilder);

  return success();
}

std::optional<linalg::LinalgOp> mlir::createAtomicLinalgGeneric(
    OpBuilder &b, Location loc, arith::AtomicRMWKind kind,
    ArrayRef<Value> inputs, ArrayRef<Value> outputs) {
  auto ctx = b.getContext();
  size_t num = inputs.size();

  // FIXME: only support all Ranks are equal now
  auto maybeRank = getRank(inputs.back());
  if (!maybeRank.has_value())
    return std::nullopt;
  auto rank = *maybeRank;

  for (auto val : inputs) {
    auto anotherMaybeRank = getRank(val);
    if (!anotherMaybeRank.has_value() || rank != *anotherMaybeRank) {
      return std::nullopt;
    }
  }

  for (auto val : outputs) {
    auto anotherMaybeRank = getRank(val);
    if (!anotherMaybeRank.has_value() || rank != *anotherMaybeRank) {
      return std::nullopt;
    }
  }

  SmallVector<AffineMap, 2> indexingMaps;

  SmallVector<utils::IteratorType, 3> parallelLoopAttrs(
      rank, utils::IteratorType::parallel);

  // insert identity map for input
  for (size_t i = 0; i < num; ++i) {
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  }

  // insert identity map for output
  for (size_t i = 0; i < num; ++i) {
    indexingMaps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  }

  ValueRange inputRange(inputs);
  ValueRange outputRange(outputs);

  linalg::LinalgOp linalgOp = b.create<linalg::GenericOp>(
      loc, inputRange, outputRange, indexingMaps, parallelLoopAttrs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        SmallVector<Value, 2> indices;
        SmallVector<Value, 2> opResults;
        // create indices
        for (unsigned i = 0; i < rank; ++i) {
          auto index = nestedBuilder.create<linalg::IndexOp>(loc, i);
          indices.push_back(index.getResult());
        }

        // create
        for (size_t i = 0; i < num; ++i) {
          auto op = nestedBuilder.create<memref::AtomicRMWOp>(
              loc, blockArgs[i].getType(), kind, blockArgs[i], outputs[i],
              indices);

          opResults.push_back(op.getResult());
        }
        nestedBuilder.create<linalg::YieldOp>(loc, opResults);
      });

  linalgOp->setAttr(
      getAtomicKindAttrName(),
      IntegerAttr::get(IntegerType::get(ctx, 32), static_cast<int64_t>(kind)));

  return linalgOp;
}
