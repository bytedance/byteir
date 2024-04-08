//===-- VectorExtTransformOps.td ------------------------------------------===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#ifndef BYTEIR_DIALECT_VECTOR_TRANSFORMOPS_VECTOR_EXT_TRANSFORMOPS
#define BYTEIR_DIALECT_VECTOR_TRANSFORMOPS_VECTOR_EXT_TRANSFORMOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ConvertReductionToGPUShuffleOp : Op<Transform_Dialect, "vector.convert_reduction_to_GPUShuffle",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Convert vector reduction op to a sequence of GPU shuffle ops.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs );

  let assemblyFormat = "$target attr-dict `:` type($target)";
}


#endif // BYTEIR_DIALECT_VECTOR_TRANSFORMOPS_VECTOR_EXT_TRANSFORMOPS