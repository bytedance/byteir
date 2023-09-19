//===-- CclTransformOps.td ------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_CCL_TRANSFORMOPS_CCLTRANSFORMOPS
#define BYTEIR_DIALECT_CCL_TRANSFORMOPS_CCLTRANSFORMOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def DecomposeAllReduceOp : Op<Transform_Dialect, "decompose_all_reduce",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    This decomposes all_reduce op into reduce_scatter and all_gather.
  }];

  let arguments = (ins 
    PDL_Operation:$target,
    I64Attr:$axis);
  
  let results = (outs 
    PDL_Operation:$reduce_scatter,
    PDL_Operation:$all_gather);

  let hasCustomAssemblyFormat = 1;
}

#endif // BYTEIR_DIALECT_CCL_TRANSFORMOPS_CCLTRANSFORMOPS