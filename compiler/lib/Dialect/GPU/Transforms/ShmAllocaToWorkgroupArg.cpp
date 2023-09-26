//===- ShmAllocaToWorkgroupArg.cpp --------------------------------- C++
//-*-===//
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

#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Transforms/MemoryPlanning.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

#define DEBUG_TYPE "shm-alloca-to-workgroup-arg"

namespace mlir {
#define GEN_PASS_DEF_SHMALLOCATOWORKGROUPARG
#include "byteir/Dialect/GPU/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;

namespace {
struct ShmAllocaToWorkgroupArgPass
    : public impl::ShmAllocaToWorkgroupArgBase<ShmAllocaToWorkgroupArgPass> {
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    WalkResult walkResult = m->walk([&](gpu::GPUFuncOp func) {
      if (!func.isKernel())
        return WalkResult::advance();

      // OpPassManager pm(func.getOperationName());
      // pm.addPass(createMemoryPlanningPass(/* alignment */ 1, /* alloca */
      // true,
      //                                     /* memory space */ 0,
      //                                     /* callback */ nullptr));
      // if (mlir::failed(runPipeline(pm, func))) {
      //   return WalkResult::interrupt();
      // }

      gpu::hoistShmAllocaToWorkgroup(func);
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      m->emitError() << "ShmAllocaToWorkgroupArgPass failed";
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::gpu::hoistShmAllocaToWorkgroup(gpu::GPUFuncOp func) {
  func->walk([&](memref::AllocaOp alloca) {
    auto memref = alloca.getType();
    if (auto memorySpace = llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
            memref.getMemorySpace())) {
      if (memorySpace.getValue() ==
          gpu::GPUDialect::getWorkgroupAddressSpace()) {
        Value workgroup = func.addWorkgroupAttribution(memref, alloca.getLoc());
        alloca.getMemref().replaceAllUsesWith(workgroup);
        alloca->erase();
      }
    }
  });
}
