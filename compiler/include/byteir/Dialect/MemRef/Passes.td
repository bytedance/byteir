//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
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


#ifndef BYTEIR_DIALECT_MEMREF_PASSES
#define BYTEIR_DIALECT_MEMREF_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Apply MemRef Affine Layout
//===----------------------------------------------------------------------===//

def ApplyMemRefAffineLayout : Pass<"apply-memref-affine-layout", "func::FuncOp"> {
  let summary = "Apply applicable MemRef layout to affineMap";
  let constructor = "mlir::createApplyMemRefAffineLayoutPass()";
}

//===----------------------------------------------------------------------===//
// RemoveCopy
//===----------------------------------------------------------------------===//

def RemoveCopy : Pass<"remove-copy", "func::FuncOp"> {
  let summary = "Remove copy";
  let constructor = "mlir::createRemoveCopyPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Simplify View
//===----------------------------------------------------------------------===//

def SimplifyView : Pass<"simplify-view", "func::FuncOp"> {
  let summary = "Simplify subview and view. Note it might create new affine.apply op if the "
                "offset of subview is dynamic.";
  let constructor = "mlir::createSimplifyViewPass()";
  let dependentDialects = [
    "affine::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExtractAddressComputationSimplify
//===----------------------------------------------------------------------===//
def ExtractAddressComputation: Pass<"extract-address-computation"> {
  let summary = "Extract address computations from memory accesses";
  let description = [{
    Extract the address computation from the instructions with memory
    accesses such that these memory accesses use only a base pointer.

    For instance,
    ```mlir
    memref.load %base[%off0, ...]
    ```

    Will be rewritten in:
    ```mlir
    %new_base = memref.subview %base[%off0,...][1,...][1,...]
    memref.load %new_base[%c0,...]
    ```
  }];
  let constructor = "mlir::createExtractAddressComputationPass()";
  let dependentDialects = [
      "memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SimplifyLinearizedIndex
//===----------------------------------------------------------------------===//
def SimplifyLinearizedIndex: Pass<"simplify-linearized-index"> {
  let summary = "Simplify linearized index";
  let constructor = "mlir::createSimplifyLinearizedIndexPass()";
  let dependentDialects = [
      "memref::MemRefDialect",
      "arith::ArithDialect"
  ];
}



#endif // BYTEIR_DIALECT_MEMREF_PASSES