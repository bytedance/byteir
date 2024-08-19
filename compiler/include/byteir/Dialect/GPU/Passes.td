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
// OptimizeVectorTransfer
//===----------------------------------------------------------------------===//
def OptimizeVectorTransfer: Pass<"optimize-vector-tranfer", "func::FuncOp"> {
  let summary = "Remove redundant transfer-write.";
  let constructor = "mlir::createOptimizeVectorTransferPass()";
}

//===----------------------------------------------------------------------===//
// GPUBlockSwizzle
//===----------------------------------------------------------------------===//
def GPUBlockSwizzle : Pass<"gpu-block-swizzle", "func::FuncOp"> {
  let summary = "Swizzle GPU block to improve performance";
  let constructor = "mlir::createGPUBlockSwizzlePass()";
  let options = [
    Option<"swizzleLogTile", "swizzle-log-tile", "int64_t", /*default=*/"0", "the log of gpu block swizzle tile">,
  ];
}

//===----------------------------------------------------------------------===//
// GPUDistributeToWarp
//===----------------------------------------------------------------------===//
def GPUDistributeToWarp: Pass<"gpu-distributed-to-warp", "func::FuncOp"> {
  let summary = "Distribute Linalg Ops to GPU warps.";
  let constructor = "mlir::createGPUDistributeToWarpPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "linalg::LinalgDialect",
    "arith::ArithDialect",
    "vector::VectorDialect",
    "gpu::GPUDialect",
  ];  
}

//===----------------------------------------------------------------------===//
// RemoveTrivialLoops
//===----------------------------------------------------------------------===//
def RemoveTrivialLoops : Pass<"remove-trivial-loops", "func::FuncOp"> {
  let summary = "Remove loops which run once in a GPU kernel.";
  let constructor = "mlir::createRemoveTrivialLoopsPass()";
}

//===----------------------------------------------------------------------===//
// GPUDistributeSharedMemoryCopy
//===----------------------------------------------------------------------===//
def GPUDistributeSharedMemoryCopy : Pass<"gpu-distribute-shared-memory-copy", "func::FuncOp"> {
  let summary = "Pass to distribute Linalg CopyOp to Vector TransferReadOp.";
  let constructor = "mlir::createGPUDistributeSharedMemoryCopyPass()";
}

//===----------------------------------------------------------------------===//
// GPUTensorCoreVectorization
//===----------------------------------------------------------------------===//
def GPUTensorCoreVectorization : Pass<"gpu-tensorcore-vectorization", "func::FuncOp"> {
  let summary = "Transform linalg.matmul to vector.contract which is compatible with MMA.";
  let constructor = "mlir::createGPUTensorCoreVectorizationPass()";
  let dependentDialects = [
      "vector::VectorDialect",
  ];
}


//===----------------------------------------------------------------------===//
// GPUPackSharedMemoryAlloc
//===----------------------------------------------------------------------===//
def GPUPackSharedMemoryAlloc : Pass<"gpu-pack-shared-memory-alloc", "func::FuncOp"> {
  let summary = "Analysis shared memory reuse and pack it into i8 alloc.";
  let constructor = "mlir::createGPUPackSharedMemoryAllocPass()";
  let dependentDialects = [
    "nvgpu::NVGPUDialect",
  ];
}

//===----------------------------------------------------------------------===//
// LegalizeGPULaunch
//===----------------------------------------------------------------------===//
def LegalizeGPULaunch : Pass<"legalize-gpu-launch", "func::FuncOp"> {
  let summary = "Legalize GPU launch ops.";
  let constructor = "mlir::createLegalizeGPULaunchPass()";
}

//===----------------------------------------------------------------------===//
// GPUInputSharedMemorySwizzle
//===----------------------------------------------------------------------===//
def GPUInputSharedMemorySwizzle: Pass<"gpu-input-shared-memory-swizzle", "func::FuncOp"> {
  let summary = "Swizzle shared memory for gemm's input to improve performance.";
  let constructor = "mlir::createGPUInputSharedMemorySwizzlePass()";
}
#endif // BYTEIR_DIALECT_GPU_PASSES
