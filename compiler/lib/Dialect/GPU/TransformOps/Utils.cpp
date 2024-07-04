//===-- Utils.cpp ------------------------------------------===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
// Some code comes from Utils.cpp for GPU transform ops in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/GPU/TransformOps/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu_ext;

/// Return a flattened thread id for the workgroup with given sizes.
template <typename ThreadOrBlockIdOp>
static Value buildLinearId(RewriterBase &rewriter, Location loc,
                           ArrayRef<OpFoldResult> originalBasisOfr) {
  assert(originalBasisOfr.size() == 3 && "expected 3 sizes");
  IndexType indexType = rewriter.getIndexType();
  AffineExpr tx, ty, tz, bdx, bdy;
  bindDims(rewriter.getContext(), tx, ty, tz);
  bindSymbols(rewriter.getContext(), bdx, bdy);
  SmallVector<OpFoldResult> vals{
      rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::x)
          .getResult(),
      rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::y)
          .getResult(),
      rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::z)
          .getResult(),
      originalBasisOfr[0], originalBasisOfr[1]};
  OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
      rewriter, loc, tx + ty * bdx + tz * bdx * bdy, vals);
  return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
}

/// Create a linear id builder that takes the `originalBasisOfr` and decompose
/// it in the basis of `forallMappingSizes`. The linear id builder returns an
/// n-D vector of ids for indexing and 1-D size + id for predicate generation.
template <typename ThreadOrBlockIdOp>
static GpuIdBuilderExtFnType commonLinearIdBuilderFn(int64_t multiplicity = 1) {
  auto res = [multiplicity](RewriterBase &rewriter, Location loc,
                            ArrayRef<OpFoldResult> forallMappingSizesOfr,
                            ArrayRef<OpFoldResult> originalBasisOfr) {
    SmallVector<Value> originalBasisValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, originalBasisOfr);
    SmallVector<Value> forallMappingSizesValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, forallMappingSizesOfr);
    SmallVector<Value> revforallMappingSizesValue(
        llvm::reverse(forallMappingSizesValue));
    OpFoldResult linearId =
        buildLinearId<ThreadOrBlockIdOp>(rewriter, loc, originalBasisOfr);
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    OpFoldResult scaledLinearId = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0.floorDiv(multiplicity), {linearId});

    SmallVector<Value> delinearIdxs(llvm::reverse(
        mlir::affine::delinearizeIndex(rewriter, loc,
                                       scaledLinearId.get<Value>(),
                                       revforallMappingSizesValue)
            .value()));

    auto activeMappingSizes = forallMappingSizesValue;
    activeMappingSizes.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, multiplicity));
    // Return n-D ids for indexing and 1-D size + id for predicate generation.
    return IdBuilderResultExt{
        /*mappingIdOps=*/delinearIdxs,
        /*availableMappingSizes=*/
        getValueOrCreateConstantIndexOp(
            rewriter, loc,
            SmallVector<OpFoldResult>{
                getIndexProduct(rewriter, loc, originalBasisValue).value()}),
        // `forallMappingSizes` iterate in the scaled basis, they need to be
        // scaled back into the original basis to provide tight
        // activeMappingSizes quantities for predication.
        /*activeMappingSizes=*/
        getValueOrCreateConstantIndexOp(
            rewriter, loc,
            SmallVector<OpFoldResult>{
                getIndexProduct(rewriter, loc, activeMappingSizes).value()}),
        /*activeIdOps=*/SmallVector<Value>{linearId.get<Value>()}};
  };

  return res;
}

/// Create a simple 3-D id builder that takes the `originalBasisOfr`
/// The 3-D id builder returns a 3-D vector of ids for indexing and 3-D sizes
/// + ids for predicate generation.
template <typename ThreadOrBlockIdOp>
static GpuIdBuilderExtFnType common3DIdBuilderFn(int64_t multiplicity = 1) {
  auto res = [multiplicity](RewriterBase &rewriter, Location loc,
                            ArrayRef<OpFoldResult> forallMappingSizesOfr,
                            ArrayRef<OpFoldResult> originalBasisOfr) {
    SmallVector<Value> forallMappingSizesValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, forallMappingSizesOfr);
    SmallVector<Value> originalBasisValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, originalBasisOfr);

    IndexType indexType = rewriter.getIndexType();
    SmallVector<Value> ids{
        rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::x),
        rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::y),
        rewriter.create<ThreadOrBlockIdOp>(loc, indexType, Dimension::z)};
    // In the 3-D mapping case, scale the first dimension by the multiplicity.
    SmallVector<Value> scaledIds = ids;
    AffineExpr d0 = getAffineDimExpr(0, rewriter.getContext());
    scaledIds[0] = affine::makeComposedFoldedAffineApply(
                       rewriter, loc, d0.floorDiv(multiplicity), {scaledIds[0]})
                       .get<Value>();
    // In the 3-D mapping case, unscale the first dimension by the multiplicity.
    auto dim0 = getConstantIntValue(forallMappingSizesOfr[0]);
    if (dim0.has_value()) {
      auto cst = rewriter.create<arith::ConstantIndexOp>(loc, multiplicity *
                                                                  dim0.value());
      forallMappingSizesValue[0] = cst;
    } else {
      auto cst = rewriter.create<arith::ConstantIndexOp>(loc, multiplicity);
      forallMappingSizesValue[0] =
          rewriter.create<arith::MulIOp>(loc, forallMappingSizesValue[0], cst);
    }
    return IdBuilderResultExt{
        /*mappingIdOps=*/scaledIds,
        /*availableMappingSizes=*/originalBasisValue,
        // `forallMappingSizes` iterate in the scaled basis, they need to be
        // scaled back into the original basis to provide tight
        // activeMappingSizes quantities for predication.
        /*activeMappingSizes=*/
        forallMappingSizesValue,
        /*activeIdOps=*/ids};
  };
  return res;
}

namespace mlir {
namespace transform {
namespace gpu_ext {

FailureOr<OpFoldResult> getIndexProduct(OpBuilder &b, Location loc,
                                        ArrayRef<Value> set) {
  if (set.empty())
    return failure();
  OpFoldResult result = set[0];
  AffineExpr s0, s1;
  bindSymbols(b.getContext(), s0, s1);
  for (unsigned i = 1, e = set.size(); i < e; i++)
    result = affine::makeComposedFoldedAffineApply(b, loc, s0 * s1,
                                                   {result, set[i]});
  return result;
}

GpuIdBuilderExt::GpuIdBuilderExt(MLIRContext *ctx, bool useLinearMapping,
                                 MappingIdBuilderExtFnType fn)
    : mappingAttributes(), idBuilder() {
  if (useLinearMapping) {
    for (uint64_t d = static_cast<uint64_t>(MappingId::LinearDim0),
                  e = getMaxEnumValForMappingId();
         d <= e; ++d)
      mappingAttributes.push_back(fn(ctx, symbolizeMappingId(d).value()));
  } else {
    for (uint64_t d = static_cast<uint64_t>(MappingId::DimX),
                  e = static_cast<uint64_t>(MappingId::DimZ);
         d <= e; ++d)
      mappingAttributes.push_back(fn(ctx, symbolizeMappingId(d).value()));
  }
}

GpuBlockIdBuilderExt::GpuBlockIdBuilderExt(MLIRContext *ctx,
                                           bool useLinearMapping)
    : GpuIdBuilderExt(ctx, useLinearMapping,
                      [](MLIRContext *ctx, MappingId id) {
                        return GPUBlockMappingAttr::get(ctx, id);
                      }) {
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<BlockIdOp>(/*multiplicity=*/1)
                  : common3DIdBuilderFn<BlockIdOp>(/*multiplicity=*/1);
}

GpuWarpgroupIdBuilderExt::GpuWarpgroupIdBuilderExt(MLIRContext *ctx,
                                                   int64_t warpSize,
                                                   bool useLinearMapping)
    : GpuIdBuilderExt(ctx, useLinearMapping,
                      [](MLIRContext *ctx, MappingId id) {
                        return GPUWarpgroupMappingAttr::get(ctx, id);
                      }),
      warpSize(warpSize) {
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<ThreadIdOp>(
                        /*multiplicity=*/kNumWarpsPerGroup * warpSize)
                  : common3DIdBuilderFn<ThreadIdOp>(
                        /*multiplicity=*/kNumWarpsPerGroup * warpSize);
}

GpuWarpIdBuilderExt::GpuWarpIdBuilderExt(MLIRContext *ctx, int64_t warpSize,
                                         bool useLinearMapping)
    : GpuIdBuilderExt(ctx, useLinearMapping,
                      [](MLIRContext *ctx, MappingId id) {
                        return GPUWarpMappingAttr::get(ctx, id);
                      }),
      warpSize(warpSize) {
  idBuilder =
      useLinearMapping
          ? commonLinearIdBuilderFn<ThreadIdOp>(/*multiplicity=*/warpSize)
          : common3DIdBuilderFn<ThreadIdOp>(/*multiplicity=*/warpSize);
}

GpuThreadIdBuilderExt::GpuThreadIdBuilderExt(MLIRContext *ctx,
                                             bool useLinearMapping)
    : GpuIdBuilderExt(ctx, useLinearMapping,
                      [](MLIRContext *ctx, MappingId id) {
                        return GPUThreadMappingAttr::get(ctx, id);
                      }) {
  idBuilder = useLinearMapping
                  ? commonLinearIdBuilderFn<ThreadIdOp>(/*multiplicity=*/1)
                  : common3DIdBuilderFn<ThreadIdOp>(/*multiplicity=*/1);
}

} // namespace gpu_ext
} // namespace transform
} // namespace mlir
