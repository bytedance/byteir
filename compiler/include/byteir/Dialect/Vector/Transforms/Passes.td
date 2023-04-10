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


#endif // BYTEIR_DIALECT_VECTOR_TRANSFORMS_PASSES