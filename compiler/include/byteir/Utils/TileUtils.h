//===- TileUtils.h --------------------------------------------*--- C++ -*-===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#ifndef BYTEIR_UTILS_TILEUTILS_H
#define BYTEIR_UTILS_TILEUTILS_H

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

SmallVector<OpFoldResult> getPadded(OpBuilder &b, ArrayRef<OpFoldResult> array,
                                    int64_t expectedSize, int64_t paddedValue);

SmallVector<OpFoldResult>
convertTileNumsToTileSizes(OpBuilder &b, Location loc,
                           ArrayRef<OpFoldResult> tileNums,
                           ArrayRef<Range> loopRanges);

SmallVector<OpFoldResult>
convertTileSizesToTileNums(OpBuilder &b, Location loc,
                           ArrayRef<OpFoldResult> tileNums,
                           ArrayRef<Range> loopRanges);

SmallVector<OpFoldResult> getValidTileNums(ArrayRef<OpFoldResult> tileNums);

SmallVector<OpFoldResult> getValidTileSizes(ArrayRef<OpFoldResult> tileSizes);

/// Options to use to control tiling.
/// It is not valid if both tileSizes and tileNums are not empty.
struct TilingOptions {
  SmallVector<int64_t> interchange = {};
  SmallVector<OpFoldResult> tileSizes = {};
  SmallVector<OpFoldResult> tileNums = {};
  SmallVector<bool> useDistributedStyle = {};

  TilingOptions &setTileSizes(ArrayRef<OpFoldResult> tileSizes) {
    this->tileSizes = llvm::to_vector(tileSizes);
    return *this;
  }
  TilingOptions &setTileNums(ArrayRef<OpFoldResult> tileNums) {
    this->tileNums = llvm::to_vector(tileNums);
    return *this;
  }
  TilingOptions &setInterchange(ArrayRef<int64_t> interchange) {
    this->interchange = llvm::to_vector(interchange);
    return *this;
  }
  TilingOptions &setUseDistributedStyle(ArrayRef<bool> useDistributedStyle) {
    this->useDistributedStyle.clear();
    for (bool v : useDistributedStyle)
      this->useDistributedStyle.push_back(v);
    return *this;
  }

  inline bool isValid() const { return isTileNums() || isTileSizes(); }
  inline bool isTileNums() const {
    return tileSizes.empty() && !tileNums.empty();
  }
  inline bool isTileSizes() const {
    return !tileSizes.empty() && tileNums.empty();
  }
};

void calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, ValueRange inductionVars,
    ArrayRef<OpFoldResult> numTiles, const SmallVector<Range> &loogRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes);

LogicalResult tileToExistedLoops(RewriterBase &rewriter, TilingInterface op,
                                 ArrayRef<OpFoldResult> tileNums,
                                 ArrayRef<int64_t> interchange,
                                 ArrayRef<bool> useDistributedStyle,
                                 scf::SCFTileAndFuseResult &tileAndFuseResult);

} // namespace mlir

#endif // BYTEIR_UTILS_TILEUTILS_H
