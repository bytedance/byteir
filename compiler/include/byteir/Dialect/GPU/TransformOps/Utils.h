//===-- Utils.h ------------------------------------------===//
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
// Some code comes from Utils.h for GPU transform ops in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_GPU_TRANSFORMOPS_UTILS_H
#define BYTEIR_DIALECT_GPU_TRANSFORMOPS_UTILS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gpu {
class GPUOp;
class LaunchOp;
enum class MappingId : uint64_t;
} // namespace gpu
namespace scf {
class ForallOp;
} // namespace scf
namespace transform {
namespace gpu_ext {

/// Create IR that computes the product of all elements in the set.
FailureOr<OpFoldResult> getIndexProduct(OpBuilder &b, Location loc,
                                        ArrayRef<Value> set);

/// Helper type for functions that generate ids for the mapping of a scf.forall.
/// Operates on both 1) an "original" basis that represents the individual
/// thread and block ids and 2) a "scaled" basis that represents grouped ids
/// (e.g. block clusters, warpgroups and warps).
/// The mapping of ids is done in the "scaled" basis (i.e. when mapping to warps
/// a division by 32 occurs).
/// The predication is in the "original" basis using the "active" quantities
/// (`activeMappingSizes`, `availableMappingSizes` and `activeIdOps`).
struct IdBuilderResultExt {
  // Ops used to replace the forall induction variables.
  SmallVector<Value> mappingIdOps;
  // Available mapping sizes used to predicate the forall body when they are
  // larger than the predicate mapping sizes.
  SmallVector<Value> availableMappingSizes;
  // Actual mapping sizes used to predicate the forall body when they are
  // smaller than the available mapping sizes.
  SmallVector<Value> activeMappingSizes;
  // Ops used to predicate the forall body when activeMappingSizes is smaller
  // than the available mapping sizes.
  SmallVector<Value> activeIdOps;
};

/// Common gpu id builder type, allows the configuration of lowering for various
/// mapping schemes. Takes:
///   - A rewriter with insertion point set before the forall op to rewrite.
///   - The loc of the forall op to rewrite.
///   - A list of positive integers carrying the mapping sizes for the current
///     forall op to rewrite.
using GpuIdBuilderExtFnType = std::function<IdBuilderResultExt(
    RewriterBase &, Location, ArrayRef<OpFoldResult>, ArrayRef<OpFoldResult>)>;

/// Helper struct for configuring the rewrite of mapped scf.forall ops to
/// various gpu id configurations.
struct GpuIdBuilderExt {
  using MappingIdBuilderExtFnType = std::function<DeviceMappingAttrInterface(
      MLIRContext *, mlir::gpu::MappingId)>;

  GpuIdBuilderExt() = default;
  GpuIdBuilderExt(MLIRContext *ctx, bool useLinearMapping,
                  MappingIdBuilderExtFnType builder);

  /// The mapping attributes targeted by this generator.
  SmallVector<DeviceMappingAttrInterface> mappingAttributes;

  /// The constructor that builds the concrete IR for mapping ids.
  GpuIdBuilderExtFnType idBuilder;
};

/// Builder for gpu::BlockIdOps used to map scf.forall to blocks.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
struct GpuBlockIdBuilderExt : public GpuIdBuilderExt {
  GpuBlockIdBuilderExt(MLIRContext *ctx, bool useLinearMapping = false);
};

/// Builder for warpgroup ids used to map scf.forall to reindexed warpgroups.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
struct GpuWarpgroupIdBuilderExt : public GpuIdBuilderExt {
  GpuWarpgroupIdBuilderExt(MLIRContext *ctx, int64_t warpSize,
                           bool useLinearMapping = false);
  int64_t warpSize = 32;
  /// In the future this may be configured by the transformation.
  static constexpr int64_t kNumWarpsPerGroup = 4;
};

/// Builder for warp ids used to map scf.forall to reindexed warps.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
struct GpuWarpIdBuilderExt : public GpuIdBuilderExt {
  GpuWarpIdBuilderExt(MLIRContext *ctx, int64_t warpSize,
                      bool useLinearMapping = false);
  int64_t warpSize = 32;
};

/// Builder for warp ids used to map scf.forall to reindexed threads.
/// If `useLinearMapping` is false, the `idBuilder` method returns 3D values
/// used for indexing rewrites as well as 3D sizes for predicate generation.
/// If `useLinearMapping` is true, the `idBuilder` method returns nD values
/// used for indexing rewrites as well as 1D sizes for predicate generation.
struct GpuThreadIdBuilderExt : public GpuIdBuilderExt {
  GpuThreadIdBuilderExt(MLIRContext *ctx, bool useLinearMapping = false);
};

} // namespace gpu_ext
} // namespace transform
} // namespace mlir

#endif // BYTEIR_DIALECT_GPU_TRANSFORMOPS_UTILS_H
