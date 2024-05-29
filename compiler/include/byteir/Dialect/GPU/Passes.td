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


#ifndef BYTEIR_DIALECT_GPU_PASSES
#define BYTEIR_DIALECT_GPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ShmAllocaToWorkgroupArg
//===----------------------------------------------------------------------===//

def ShmAllocaToWorkgroupArg : Pass<"shm-alloca-to-workgroup-arg", "gpu::GPUModuleOp"> {
  let summary = "Hoist shared memory alloca in gpu kernel to workgroup argument";
  let dependentDialects = [
    "gpu::GPUDialect",
    "memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GPUBlockSwizzle
//===----------------------------------------------------------------------===//
def GPUBlockSwizzle : Pass<"gpu-block-swizzle", "func::FuncOp"> {
  let summary = "Swizzle GPU block to improve performance";
  let constructor = "mlir::createGPUBlockSwizzlePass()";
  let options = [
    Option<"swizzleLogTile", "swizzle_log_tile", "int", /*default=*/"0", "the log of gpu block swizzle tile">
  ];
}

#endif // BYTEIR_DIALECT_GPU_PASSES
