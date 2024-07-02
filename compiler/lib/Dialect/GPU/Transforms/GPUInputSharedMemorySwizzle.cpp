//===- GPUInputSharedMemorySwizzle.cpp -------------------------*--- C++-*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "byteir/Dialect/GPU/Transforms/GPUInputSharedMemorySwizzle.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {

static void swizzleSharedMemory(scf::ForallOp forallOp) {
  SmallVector<memref::AllocOp> shmAllocOps;
  forallOp->walk([&](memref::AllocOp allocOp) {
    // Only apply it to shared memory of input operands.
    if (!nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(allocOp.getType())) {
      return;
    }
    if (hasMarker(allocOp, {getAllocSharedMemoryAMarker(),
                            getAllocSharedMemoryBMarker()})) {
      shmAllocOps.push_back(allocOp);
    }
  });
  for (auto allocOp : shmAllocOps) {
    (void)nvgpu::optimizeSharedMemoryReadsAndWrites(forallOp,
                                                    allocOp.getMemref());
  }
}

struct GPUInputSharedMemorySwizzlePass
    : public GPUInputSharedMemorySwizzleBase<GPUInputSharedMemorySwizzlePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getBody());

    if (!hasGemmTileConfig(funcOp)) {
      return;
    }

    auto forallOpOptional = getForallOpMappedTo2DBlock(funcOp);
    if (!forallOpOptional.has_value()) {
      return signalPassFailure();
    }
    scf::ForallOp forallOp = *forallOpOptional;
    swizzleSharedMemory(forallOp);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUInputSharedMemorySwizzlePass() {
  return std::make_unique<GPUInputSharedMemorySwizzlePass>();
}