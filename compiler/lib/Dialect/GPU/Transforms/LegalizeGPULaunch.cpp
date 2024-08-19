//===- LegalizeGPULaunch.cpp --------------------------------------------*-===//
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

#include "byteir/Dialect/GPU/Transforms/LegalizeGPULaunch.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include <string>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {

static int64_t getSharedMemorySizeInGPULaunch(gpu::LaunchOp op) {
  int64_t sharedMemSizeInBytes = 0;
  op->walk([&](memref::AllocaOp allocaOp) {
    if (nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(allocaOp.getType())) {

      sharedMemSizeInBytes +=
          allocaOp.getType().getNumElements() *
          allocaOp.getType().getElementType().getIntOrFloatBitWidth() / 8;
    }
  });
  op->walk([&](memref::AllocOp allocOp) {
    if (nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(allocOp.getType())) {
      sharedMemSizeInBytes +=
          allocOp.getType().getNumElements() *
          allocOp.getType().getElementType().getIntOrFloatBitWidth() / 8;
    }
  });
  return sharedMemSizeInBytes;
}

struct LegalizeGPULaunchPass
    : public LegalizeGPULaunchBase<LegalizeGPULaunchPass> {
  LegalizeGPULaunchPass() : LegalizeGPULaunchBase<LegalizeGPULaunchPass>() {}
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    auto launchOps = funcOp.getOps<gpu::LaunchOp>();
    for (auto launchOp : launchOps) {
      int64_t sharedMemSize = getSharedMemorySizeInGPULaunch(launchOp);
      if (sharedMemSize < 48 * 1024) // 48kB
        continue;
      builder.setInsertionPoint(launchOp);
      Value sharedMemSizeValue = builder.create<arith::ConstantOp>(
          launchOp.getLoc(), builder.getI32IntegerAttr(sharedMemSize));
      if (!launchOp.getDynamicSharedMemorySizeMutable().empty()) {
        continue;
      }
      launchOp.getDynamicSharedMemorySizeMutable().append(
          ValueRange{sharedMemSizeValue});
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLegalizeGPULaunchPass() {
  return std::make_unique<LegalizeGPULaunchPass>();
}
