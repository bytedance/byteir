//===-- GPUExtTransformOps.td ------------------------------------------===//
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
// Some code comes from GPUTransformOps.td in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_GPU_TRANSFORMOPS_GPU_EXT_TRANSFORMOPS
#define BYTEIR_DIALECT_GPU_TRANSFORMOPS_GPU_EXT_TRANSFORMOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"


def MapNestedForallToThreadsExtOp :
  Op<Transform_Dialect, "gpu.map_nested_forall_to_threads_ext",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
      This operation is extended from gpu.map_nested_forall_to_threads to
      support `scf.forall` with dynamic trip counts.

      Target the `gpu.launch op` and rewrite all `scf.forall` nested in it to 
      distributed `gpu.thread_id` attribute.

      The operation searches for `scf.forall` ops nested under `target` and maps
      each such op to GPU threads. 
      
      `scf.forall` induction variables are rewritten to `gpu.thread_id` according
      to the `mapping` attribute.

      Different types of mappings attributes are supported:
        - the block_dims is a list of integers that specifies the number of
          threads in each dimension. This is a mandatory attribute that is used
          to constrain the number of threads in each dimension. If an 
          `scf.forall` op is mapped to fewer threads, predication occurs.
        - the warp_dims is a list of integers that specifies the number of
          warps in each dimension. This is an optional attribute that is used
          to constrain the number of warps in each dimension. When present, this
          attribute must be specified in a way that is compatible with the 
          block_dims attribute. If an `scf.forall` op is mapped to fewer warps,
          predication occurs.

      Dynamic `scf.forall` trip counts are currently supported.
      Dynamic block dim sizes are currently not supported.

      Only **bufferized** `scf.forall` are currently supported.
      Only `scf.forall` distributed to **at most 3 dimensions** are
      currently supported.

      The `sync_after_distribute`attribute controls whether a `gpu.barrier` is
      inserted after each scf.forall op. At this time, this is an all or nothing
      choice. This will need to be tightened in the future.

      The operation alters the block size of the given gpu_launch using the 
      mandatory block_dims argument.

      #### Return modes:

      This operation ignores non-gpu_launch ops and drops them in the return.

      If any scf.forall with tensors is found, the transform definitely
      fails.

      If all the scf.forall operations with gpu.thread mapping contained
      within the LaunchOp referred to by the `target` PDLOperation lower to GPU
      properly, the transform succeeds. Otherwise the transform definitely
      fails.

      scf.forall operations with mappings other than gpu.thread are
      ignored.

      The returned handle points to the same LaunchOp operand, consuming it and
      producing a new SSA value to satisfy chaining and linearity of the IR
      properties.
    }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DenseI64ArrayAttr: $block_dims,
                   DefaultValuedAttr<BoolAttr, "true">:$sync_after_distribute,
                   DefaultValuedAttr<I64Attr, "32">:$warp_size);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $target
    `block_dims` `=` $block_dims
    (`sync_after_distribute` `=` $sync_after_distribute^)?
    (`warp_size` `=` $warp_size^)?
    attr-dict
    `:` functional-type($target, $result)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];

  let hasVerifier = 1;
}

def MapForallToBlocksExtOp :
  Op<Transform_Dialect, "gpu.map_forall_to_blocks_ext",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    This operation is extended from gpu.map_forall_to_blocks to
    support `scf.forall` with dynamic trip counts.

    The grid dims are expected to be provided in the `grid_dims` attribute.
    When the dims are not known statically, the corresponding entry in
    the `grid_dims` attribute must be set to ShapedType::kDynamic and will
    infered from `scf.forall` automatically. For common 3D mapping, the entry
    will be set to the corresponding trip count in `scf.forall`. For linear mapping,
    grid_dims will be set to (total_trip_count, 1, 1).

    Target the gpu_launch op and rewrite the top level `scf.forall`
    to distributed gpu.block_id attribute. If `generate_gpu_launch` attribute
    is set, then first generates `gpu_launch` and moves the top level
    `scf.forall` inside.

    The operation searches top level `scf.forall` ops under
    `gpu_launch` and maps each such op to GPU blocks. Mapping is
    one-to-one and the induction variables of `scf.forall` are
    rewritten to gpu.block_id according to the `thread_dim_mapping` attribute.

    Dynamic `scf.forall` trip counts are currently supported.
    Dynamic grid dim sizes are currently supported.

    Only **bufferized** scf.forall are currently supported.
    Only scf.forall distributed to **at most 3 dimensions** are
    currently supported.

    The operation alters the grid size of the given gpu_launch using the
    grid_dims argument.

    #### Return modes:

    This operation ignores non-gpu_launch ops and drops them in the return.

    If any scf.forall with tensors is found, the transform definitely
    fails.

    If all the scf.forall operations contained within the LaunchOp
    referred to by the `target` PDLOperation lower to GPU properly, the
    transform succeeds. Otherwise the transform definitely fails.

    The returned handle points to the same LaunchOp operand, consuming it and
    producing a new SSA value to satisfy chaining and linearity of the IR
    properties.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$grid_dims,
                   UnitAttr:$generate_gpu_launch);
  let results = (outs TransformHandleTypeInterface:$result);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];

  let hasCustomAssemblyFormat = 1;

}

#endif // BYTEIR_DIALECT_GPU_TRANSFORMOPS_GPU_EXT_TRANSFORMOPS
