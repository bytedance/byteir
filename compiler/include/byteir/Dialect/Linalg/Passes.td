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


#ifndef BYTEIR_DIALECT_LINALG_PASSES
#define BYTEIR_DIALECT_LINALG_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LinalgExtBufferize
//===----------------------------------------------------------------------===//

def LinalgExtBufferize : Pass<"linalg-ext-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the linalg-ext and linalg dialect";
  let constructor = "mlir::createLinalgExtBufferizePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "linalg_ext::LinalgExtDialect",
    "memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// LinalgExtLowerToLoops
//===----------------------------------------------------------------------===//

def LinalgExtLowerToLoops : Pass<"convert-linalg-ext-to-loops", "func::FuncOp"> {
  let summary = "Lower the operations from the linalg ext dialect into loops";
  let constructor = "mlir::createConvertLinalgExtToLoopsPass()";
  let dependentDialects = [
    "linalg_ext::LinalgExtDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "affine::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LinalgDataPlace
//===----------------------------------------------------------------------===//

def LinalgDataPlace : Pass<"linalg-data-place", "func::FuncOp"> {
  let summary = "Data-Place Linalg's buffer to a local buffer with space";
  let constructor = "mlir::createLinalgDataPlacePass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "arith::ArithDialect",
  ];
  let options = [
    ListOption<"memSpaces", "mem-spaces", "int64_t",
               "data placement memory spaces",
               "llvm::cl::ZeroOrMore">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgElementwiseFusionExt
//===----------------------------------------------------------------------===//

def LinalgElementwiseFusionExt : Pass<"linalg-fuse-elementwise-ext"> {
  let summary = "Extension of elementwise fusion on tensors";
  let constructor = "mlir::createLinalgElementwiseFusionExtPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "linalg_ext::LinalgExtDialect",
    "memref::MemRefDialect"
  ];
  let options = [
    Option<"enableSharedInput", "shared-input", "bool", /*default=*/"false",
           "enable shared input fusion">,
    Option<"enableDiffShapes", "diff-shapes", "bool", /*default=*/"false",
           "enable outputs with different shapes">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgOpTiling
//===----------------------------------------------------------------------===//

def LinalgOpTiling : Pass<"linalg-op-tile", "func::FuncOp"> {
  let summary = "Tile operations in the linalg and linalg-ext dialect";
  let constructor = "mlir::createLinalgOpTilingPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "linalg_ext::LinalgExtDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"loopType", "loop-type", "std::string", /*default=*/"\"scf\"",
           "Specify the type of loops to generate: for, parallel">
  ];
}

//===----------------------------------------------------------------------===//
// LinalgPrefetch
//===----------------------------------------------------------------------===//

def LinalgPrefetch : Pass<"linalg-prefetch", "func::FuncOp"> {
  let summary = "Prefetch Linalg's copies within loop";
  let constructor = "mlir::createLinalgPrefetchPass()";
  let options = [
    Option<"prefetchCnt", "count", "int64_t", /*default=*/"1",
           "prefetch count, <= 0 implies disabled">,
    Option<"unroll", "unroll", "bool", /*default=*/"false",
           "unroll prefetch">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgScopeTiling
//===----------------------------------------------------------------------===//

def LinalgScopeTiling : Pass<"linalg-scope-tile", "func::FuncOp"> {
  let summary = "Perform Tile linalg operations within a scope";
  let constructor = "mlir::createLinalgScopeTilingPass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    Option<"tileAxis", "axis", "int64_t", /*default=*/"0",
           "tile axis">,
    Option<"tileSize", "tile-size", "int64_t", /*default=*/"0",
           "tile size">,
    Option<"parallelizeReduction", "par-reduce", "bool", /*default=*/"false",
            "force parallelize reduction">,
    Option<"loopType", "loop-type", "std::string", /*default=*/"\"scf\"",
           "Specify the type of loops to generate: scf, affine, tiled_loop">,
    Option<"keepTag", "keep-tag", "bool", /*default=*/"false",
           "keep tag">,
  ];
}

//===----------------------------------------------------------------------===//
// LinalgCollapseLoops
//===----------------------------------------------------------------------===//

def LinalgCollapseLoops : Pass<"linalg-collapse-loops", "func::FuncOp"> {
  let summary = "Collapse loops on linalg.generic ops";
  let constructor = "mlir::createLinalgCollapseLoops()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LinalgGeneralizationExt
//===----------------------------------------------------------------------===//

def LinalgGeneralizationExt : Pass<"linalg-generalization-ext", "func::FuncOp"> {
  let summary = "Generalize linalg ops into linalg.generic";
  let dependentDialects = [
    "linalg::LinalgDialect"
  ];
}

#endif // BYTEIR_DIALECT_LINALG_PASSES