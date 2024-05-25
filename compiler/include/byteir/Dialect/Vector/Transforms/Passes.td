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


#ifndef BYTEIR_DIALECT_VECTOR_TRANSFORMS_PASSES
#define BYTEIR_DIALECT_VECTOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Vector transpose lowering
//===----------------------------------------------------------------------===//

def VectorTransposeLoweringPass : Pass<"vector-transpose-lowering", "func::FuncOp"> {
  let summary = "Pass to lower vector.transpose";
  let dependentDialects = [
    "memref::MemRefDialect",
    "vector::VectorDialect"
  ];
  let options = [
      Option<"enableAVX2", "enable-avx2", "bool", "true", "Whether to use avx2 intrinsincs">
  ];
}

//===----------------------------------------------------------------------===//
// Move Forall Region Into WarpOp
//===----------------------------------------------------------------------===//

def MoveForallRegionIntoWarpOpPass : Pass<"move-forall-region-into-warp-op", "mlir::func::FuncOp"> {
  let summary = "move region of forall into warp_execute_on_lane_0 op";
  let constructor = "mlir::createMoveForallRegionIntoWarpOpPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "vector::VectorDialect",
    "gpu::GPUDialect",
  ];
  let options = [
    Option<"warpSize", "warp-size", "int64_t", "32", "warp size">,
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

//===----------------------------------------------------------------------===//
// Vector Warp Distribute
//===----------------------------------------------------------------------===//

def VectorWarpDistributePass : Pass<"vector-warp-distribute", "mlir::func::FuncOp"> {
  let summary = "vector warp distribute transformation";
  let constructor = "mlir::createVectorWarpDistributePass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "vector::VectorDialect",
    "gpu::GPUDialect",
    "affine::AffineDialect",
  ];
  let options = [
    Option<"warpOpToSCF", "rewrite-warp-ops-to-scf-if", "bool",
          /*default=*/"false",
          "Lower vector.warp_execute_on_lane0 to scf.if op">,

    Option<"distributeTransferWriteOps", "distribute-transfer-write", "bool",
          /*default=*/"false",
          "distribution of transfer write">,

    Option<"hoistUniform", "hoist-uniform", "bool",
          /*default=*/"false",
          "hoist-uniform">,

    Option<"propagateDistribution", "propagate-distribution", "bool",
          /*default=*/"false",
          "distribution propgation">,

    Option<"maxTransferWriteElements", "max-transfer-write-elements", "int64_t",
          /*default=*/"1",
          "Maximum number of transfer write elements to distribute">,
  ];
}

//===----------------------------------------------------------------------===//
// Scalar Vector Lowering
//===----------------------------------------------------------------------===//

def ScalarVectorLoweringPass : Pass<"scalar-vector-lowering", "func::FuncOp"> {
  let summary = "Pass to lower scalar vector";
  let dependentDialects = [
    "memref::MemRefDialect",
    "vector::VectorDialect"
  ];
}
#endif // BYTEIR_DIALECT_VECTOR_TRANSFORMS_PASSES