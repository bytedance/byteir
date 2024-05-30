//===- GPUBlockSwizzle.cpp ------------------------------ C++-*-===//
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

#include "byteir/Dialect/GPU/Transforms/GPUBlockSwizzle.h"
#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Transforms/MemoryPlanning.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

#include "PassDetails.h"

#define DEBUG_TYPE "gpu-block-swizzle"

using namespace llvm;
using namespace mlir;

namespace {
bool isMappedToGPUBlocks(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUBlockMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

/// Implements the following swizzling logic:
/// void getTiledId2(unsigned x, unsigned y, unsigned* tiledx,
///                  unsigned* tiledy) {
///  unsigned t_tiledx = (x + (y % tile) * grid_size_x) / tile;
///  unsigned t_tiledy = (y / tile) * tile +
///      (x + (y % tile) * grid_size_x) % tile;
///  bool c = grid_size_y % tile != 0 &&
///      ((y / tile) * tile + tile) > grid_size_y;
///  *tiledx = c ? x : t_tiledx;
///  *tiledy = c ? y : t_tiledy;
/// }
static std::pair<Value, Value>
makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
                Value workgroupIdY, Value workgroupCountX,
                Value workgroupCountY, unsigned swizzleTile) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value tile = b.create<arith::ConstantIndexOp>(loc, swizzleTile);
  Value yModTile = b.create<arith::RemUIOp>(loc, workgroupIdY, tile);
  Value yDivTile = b.create<arith::DivUIOp>(loc, workgroupIdY, tile);
  Value swizzleParam = b.create<arith::MulIOp>(loc, yModTile, workgroupCountX);
  Value swizzleParam2 =
      b.create<arith::AddIOp>(loc, workgroupIdX, swizzleParam);
  Value swizzleParam3 = b.create<arith::RemUIOp>(loc, swizzleParam2, tile);
  Value swizzleParam4 = b.create<arith::MulIOp>(loc, yDivTile, tile);
  Value unboundedSwizzledIdX =
      b.create<arith::DivUIOp>(loc, swizzleParam2, tile);
  Value unboundedSwizzledIdY =
      b.create<arith::AddIOp>(loc, swizzleParam3, swizzleParam4);
  Value gyModTile = b.create<arith::RemUIOp>(loc, workgroupCountY, tile);
  Value gyAddTile = b.create<arith::AddIOp>(loc, swizzleParam4, tile);
  Value condition1 =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, gyModTile, zero);
  Value condition2 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             gyAddTile, workgroupCountY);
  Value condition3 = b.create<arith::AndIOp>(loc, condition1, condition2);
  Value swizzledIdX = b.create<arith::SelectOp>(loc, condition3, workgroupIdX,
                                                unboundedSwizzledIdX);
  Value swizzledIdY = b.create<arith::SelectOp>(loc, condition3, workgroupIdY,
                                                unboundedSwizzledIdY);
  return {swizzledIdX, swizzledIdY};
}

// Only support 2d grid.
static LogicalResult reorderForallOpInFunc(func::FuncOp func,
                                           unsigned swizzleLogTile) {
  llvm::errs() << "swizzleLogTile: " << swizzleLogTile << "\n";
  unsigned swizzleTile = 1 << swizzleLogTile;
  std::vector<scf::ForallOp> forallOps;
  func.walk([&](scf::ForallOp forallOp) {
    if (isMappedToGPUBlocks(forallOp) &&
        forallOp.getMappingAttr().getValue().size() == 2)
      forallOps.push_back(forallOp);
  });
  if (forallOps.size() != 1)
    return failure();
  scf::ForallOp forallOp = forallOps[0];

  OpBuilder b(forallOp);

  scf::ForallOp newforallOp = forallOp.clone();
  newforallOp.getBody()->clear();
  b.insert(newforallOp);

  // This way will not copy attributes.
  // scf::ForallOp newforallOp = b.create<scf::ForallOp>(
  //     forallOp.getLoc(), forallOp.getMixedLowerBound(),
  //     forallOp.getMixedUpperBound(), forallOp.getMixedStep(),
  //     forallOp.getResults(), forallOp.getMapping());

  b.setInsertionPointToStart(newforallOp.getBody());
  auto originLoops = forallOp.getInductionVars();
  auto gridSize = newforallOp.getUpperBound(b);
  auto loops = newforallOp.getInductionVars();
  auto mapping = newforallOp.getMappingAttr().getValue();

  Value workgroupIdX =
      loops[mapping[0].cast<gpu::GPUBlockMappingAttr>().getMappingId()];
  Value workgroupIdY =
      loops[mapping[1].cast<gpu::GPUBlockMappingAttr>().getMappingId()];
  Value workgroupCountX =
      gridSize[mapping[0].cast<gpu::GPUBlockMappingAttr>().getMappingId()];
  Value workgroupCountY =
      gridSize[mapping[1].cast<gpu::GPUBlockMappingAttr>().getMappingId()];

  auto [swizzledIdX, swizzledIdY] =
      makeSwizzledIds(newforallOp.getLoc(), b, workgroupIdX, workgroupIdY,
                      workgroupCountX, workgroupCountY, swizzleTile);

  IRMapping bvm;
  bvm.map(originLoops[0], swizzledIdX);
  bvm.map(originLoops[1], swizzledIdY);
  for (auto &op : forallOp.getBody()->getOperations()) {
    b.clone(op, bvm);
  }
  forallOp.replaceAllUsesWith(newforallOp);
  forallOp.erase();
  return success();
}

struct GPUBlockSwizzlePass : public GPUBlockSwizzleBase<GPUBlockSwizzlePass> {
public:
  GPUBlockSwizzlePass(int64_t swizzleLogTile) : GPUBlockSwizzleBase() {
    this->swizzleLogTile = swizzleLogTile;
  }

  void runOnOperation() override {
    func::FuncOp op = getOperation();
    if (!op->hasAttr("__byteir_matmul_epilogue_fusion__"))
      return;
    if (failed(reorderForallOpInFunc(op, swizzleLogTile))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUBlockSwizzlePass(int64_t swizzleLogTile) {
  return std::make_unique<GPUBlockSwizzlePass>(swizzleLogTile);
}
