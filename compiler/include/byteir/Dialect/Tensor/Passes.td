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

#ifndef BYTEIR_DIALECT_TENSOR_PASSES
#define BYTEIR_DIALECT_TENSOR_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TensorPadSpecialization
//===----------------------------------------------------------------------===//

def TensorPadSpecialization : Pass<"tensor-pad-specialization", ""> {
  let summary = "Specialize tensor.pad op";
  let constructor = "mlir::createTensorPadSpecializationPass()";
  let dependentDialects = [
    "scf::SCFDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ExtractSliceSpecialization
//===----------------------------------------------------------------------===//

def ExtractSliceSpecialization : Pass<"extract-slice", ""> {
  let summary = "Specialize for extract slice Op";
  let constructor = "mlir::createExtractSliceSpecializationPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "linalg::LinalgDialect",
  ];
}

#endif // BYTEIR_DIALECT_TENSOR_PASSES
