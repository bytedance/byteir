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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#define DEBUG_TYPE "gpu-pipelining"

using namespace mlir;

namespace {

/// Helper to recursively add operation dependencies within `block` to `dep`
/// set.
static void addDepOps(llvm::SmallDenseSet<Operation *> &dep, Operation *op,
                      Block *block) {
  if (!dep.insert(op).second)
    return;
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block)
      addDepOps(dep, defOp, block);
  }
}

static void
getPipelineStages(scf::ForOp forOp,
                  std::vector<std::pair<Operation *, unsigned>> &ops,
                  unsigned depth) {
  SmallVector<linalg::CopyOp> copyOps;
  forOp.walk([&](linalg::CopyOp copyOp) {
    if (hasMarker(copyOp, {getCopyToSharedMemoryAMarker(),
                           getCopyToSharedMemoryBMarker()})) {
      copyOps.push_back(copyOp);
    }
  });

  llvm::SmallDenseSet<Operation *> loadDep;
  for (linalg::CopyOp copyOp : copyOps) {
    addDepOps(loadDep, copyOp, forOp.getBody());
  }

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (!loadDep.count(&op) && !isa<scf::YieldOp>(op))
      ops.push_back(std::make_pair(&op, depth));
  }
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (loadDep.count(&op))
      ops.push_back(std::make_pair(&op, 0));
  }
}

static Operation *replaceOpWithPredicatedOp(RewriterBase &rewriter,
                                            Operation *op, Value pred) {
  Location loc = op->getLoc();
  if (!isa<linalg::CopyOp>(op))
    return op;
  auto ifOp = rewriter.create<scf::IfOp>(loc, op->getResultTypes(), pred, true);
  // True branch.
  op->moveBefore(&ifOp.getThenRegion().front(),
                 ifOp.getThenRegion().front().begin());
  rewriter.setInsertionPointAfter(op);
  if (op->getNumResults() > 0)
    rewriter.create<scf::YieldOp>(loc, op->getResults());
  return ifOp.getOperation();
}

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

    // step 2: find linalg.copy ops in scf.for and its dependencies
    SmallVector<scf::ForOp> forOps;
    // Mark the loop with shared memory copy for pipelining.
    funcOp.walk([&forOps](scf::ForOp forOp) { forOps.push_back(forOp); });

    assert(forOps.size() == 1 && "Only support 1 loop in matmul");

    scf::PipeliningOption options;
    unsigned maxDepth = stages;
    auto getSchedule =
        [maxDepth](scf::ForOp forOp,
                   std::vector<std::pair<Operation *, unsigned>> &schedule) {
          getPipelineStages(forOp, schedule, maxDepth);
        };

    // step 3: apply software pipelining
    options.getScheduleFn = getSchedule;
    options.supportDynamicLoops = false;
    options.peelEpilogue = false;
    options.predicateFn = replaceOpWithPredicatedOp;

    RewritePatternSet patterns(&getContext());
    scf::populateSCFLoopPipeliningPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUPipeliningPass(int64_t stages) {
  return std::make_unique<GPUPipeliningPass>(stages);
}