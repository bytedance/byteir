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


#ifndef BYTEIR_DIALECT_AFFINE_PASSES
#define BYTEIR_DIALECT_AFFINE_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// RewriteAffineToMemref
//===----------------------------------------------------------------------===//

def RewriteAffineToMemref : Pass<"rewrite-affine-to-memref", "func::FuncOp"> {
  let summary = "Rewrite Affine Load/Store to Memref Load/Store";
  let constructor = "mlir::createRewriteAffineToMemrefPass()";
  let dependentDialects = [
    "memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// InsertTrivialAffineLoop
//===----------------------------------------------------------------------===//

def InsertTrivialAffineLoop : Pass<"insert-trivial-affine-loop", "func::FuncOp"> {
  let summary = "Insert trivial affine Loop for scalar computation";
  let constructor = "mlir::createInsertTrivialAffineLoopPass()";
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

//===----------------------------------------------------------------------===//
// AffineLoopFusionEx
//===----------------------------------------------------------------------===//

def AffineLoopFusionEx : Pass<"affine-loop-fusion-ex", "func::FuncOp"> {
  let summary = "Affine Loop Fusion Extension";
  let constructor = "mlir::createAffineLoopFusionExPass()";
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

#endif // BYTEIR_DIALECT_AFFINE_PASSES