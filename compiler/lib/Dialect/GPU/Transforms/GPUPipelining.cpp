//===- GPUPipelining.cpp -------------------------------------*--- C++-*-===//
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

#include "byteir/Dialect/GPU/Transforms/GPUPipelining.h"
#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "byteir/Dialect/MemRef/Transforms/MultiBufferExt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#define DEBUG_TYPE "gpu-pipelining"

using namespace mlir;

namespace {
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

// static void
// getSchedule(scf::ForOp forOp,
//             std::vector<std::pair<Operation *, unsigned>> &schedule) {
//   if (!forOp->hasAttr(kTestPipeliningLoopMarker))
//     return;

//   schedule.resize(forOp.getBody()->getOperations().size() - 1);
//   forOp.walk([&schedule](Operation *op) {
//     auto attrStage =
//     op->getAttrOfType<IntegerAttr>(kTestPipeliningStageMarker); auto
//     attrCycle =
//         op->getAttrOfType<IntegerAttr>(kTestPipeliningOpOrderMarker);
//     if (attrCycle && attrStage) {
//       // TODO: Index can be out-of-bounds if ops of the loop body disappear
//       // due to folding.
//       schedule[attrCycle.getInt()] =
//           std::make_pair(op, unsigned(attrStage.getInt()));
//     }
//   });
// }

struct GPUPipeliningPass : public GPUPipeliningBase<GPUPipeliningPass> {
  GPUPipeliningPass(int64_t stages) : GPUPipeliningBase() {
    this->stages = stages;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    // step 1: collect all the alloc operations and do multi-buffering
    SmallVector<memref::AllocaOp> allocas;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocaOp allocaOp) {
      if (nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(
              allocaOp.getType()) &&
          hasMarker(allocaOp, {getAllocSharedMemoryAMarker(),
                               getAllocSharedMemoryBMarker()})) {
        allocas.push_back(allocaOp);
      }
    });
    assert(allocas.size() == 2 && "Only support 2 allocas for now");
    // Apply multi-buffering to all of them.
    for (memref::AllocaOp allocaOp : allocas) {
      if (failed(memref::multiBufferExt(allocaOp, (unsigned int)stages))) {
        // Error out and stop if any buffer cannot be multi buffered, as
        // future software pipelining transformations will assume this
        // happened.
        allocaOp.emitOpError("cannot be multi-buffered");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUPipeliningPass(int64_t stages) {
  return std::make_unique<GPUPipeliningPass>(stages);
}