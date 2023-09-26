//===-- LinalgExtTransformOps.td ------------------------------------------===//
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
// Some code comes from LinalgTransformOps.td in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMOPS_LINALG_EXT_TRANSFORMOPS
#define BYTEIR_DIALECT_LINALG_TRANSFORMOPS_LINALG_EXT_TRANSFORMOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


def AnnotateExtOp : Op<Transform_Dialect, "annotate_ext",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    this is for annotation.
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
}

def CollapseDimsOp : Op<Transform_Dialect, "structured.collapse_dims",
       [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Collapse dimensions of linalg.generic operation
  }];

  let arguments = (ins PDL_Operation:$target,
                       IndexListArrayAttr:$reassociation);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target $reassociation attr-dict";
  
  code extraClassDeclaration = [{
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : getReassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(::llvm::cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
              return ::llvm::cast<IntegerAttr>(indexAttr).getInt();
            })));
      return reassociationIndices;
    };
  }];
}

def DetensorizeOp : Op<Transform_Dialect, "structured.detensorize",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Detensorize linalg ops.
  }];

  let arguments = (ins PDL_Operation:$target);

  let assemblyFormat = "$target attr-dict";
}

def FoldUnitExtentDimsOp : Op<Transform_Dialect, "structured.fold_unit_extent_dims",
       [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Remove unit-extent dimension in Linalg ops on tensors
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
}

def FuseExtOp : Op<Transform_Dialect, "structured.fuse_ext",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Tiles the operations pointed to by the target handle and fuses their
    producers greedily using the options provided as attributes.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         Optional<PDL_Operation>:$stop,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_interchange,
         DefaultValuedAttr<BoolAttr, "false">:$keep_intermediates);
  let results = (outs PDL_Operation:$transformed,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LowerToLoopsOp : Op<Transform_Dialect, "structured.lower_to_loops",
       [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Lowers the operations to loops.
  }];

  let arguments = (ins PDL_Operation:$target, 
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$loop_ids);
  let results = (outs Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;
}

def InlineOp : Op<Transform_Dialect, "structured.inline",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    inline function call
  }];

  let arguments = (ins PDL_Operation:$target);

  let assemblyFormat = "$target attr-dict";
}


def LinalgOutlineOp : Op<Transform_Dialect, "structured.outline",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Outline the linalg operation to a named function, where `libcall` for
    the external library call.

    If `libcall` was set to False, every outlined function would have a
    unique name , in this situation `func_name` just gives a naming hint.
    Otherwise, all transformed function calls refer to the same external
    function named `func_name`.
  }];

  let arguments =
    (ins PDL_Operation:$target, StrAttr:$func_name,
         UnitAttr:$libcall);
  let results = (outs PDL_Operation:$functions,
                      PDL_Operation:$calls);

  let builders = [
    OpBuilder<(ins "Value":$target, "StringRef":$func_name, "bool":$libcall), [{
      $_state.addAttribute(LinalgOutlineOp::getFuncNameAttrName($_state.name),
                           $_builder.getStringAttr(func_name));
      if (libcall) {
        $_state.addAttribute(LinalgOutlineOp::getLibcallAttrName($_state.name),
                             $_builder.getUnitAttr());
      }
      Type pdlType = pdl::OperationType::get($_builder.getContext());
      $_state.addTypes({pdlType, pdlType});
      $_state.addOperands(target);
    }]>
  ];

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def TileLoopHintOp : Op<Transform_Dialect, "structured.tile_loop_hint",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    this is a tile annotation labeling whether a loop parallel
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let assemblyFormat = "$target attr-dict `:` type($target)";
}

def TileExtOp : Op<Transform_Dialect, "structured.tile_ext",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    this is a tile extension
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$dynamic_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$interchange);
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

def SharedOutputToDistributedStyleOp : Op<Transform_Dialect, 
        "structured.shared_output_to_distributed_style",
        [DeclareOpInterfaceMethods<TransformOpInterface>,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
      Convert parallel tiling from shared output style to distributed style.
  }];

  let arguments = (ins 
      PDL_Operation:$loop, 
      PDL_Operation:$init, 
      PDL_Operation:$merge);

  let results = (outs PDL_Operation:$new_loop,
                      PDL_Operation:$new_init);

  let hasCustomAssemblyFormat = 1;
}

def FuseOperandsOp : Op<Transform_Dialect, "structured.fuse_operands",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Tiles all the operands of the target handle together and fuses
    their producers greedily using the options provided as attributes.
  }];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_nums,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_interchange,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$use_distributed,
         DefaultValuedAttr<BoolAttr, "false">:$expect_whole_graph_fusion);
  let results = (outs PDL_Operation:$transformed,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


def InsertSliceToCopyExtOp :
  Op<Transform_Dialect, "structured.insert_slice_to_copy_ext",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let description = [{
    insert_slice_to_copy_ext extension.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results) ";

  let builders = [
    OpBuilder<(ins "Value":$target)>,
  ];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}


#endif // BYTEIR_DIALECT_LINALG_TRANSFORMOPS_LINALG_EXT_TRANSFORMOPS