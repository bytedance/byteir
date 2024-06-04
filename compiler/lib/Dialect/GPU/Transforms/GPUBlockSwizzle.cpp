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
// Some code comes from
// compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp of
// IREE project.
// Original license:
// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "byteir/Dialect/GPU/Transforms/GPUBlockSwizzle.h"
#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

#include "PassDetail.h"

#define DEBUG_TYPE "gpu-block-swizzle"

using namespace llvm;
using namespace mlir;

namespace {

// Implements the following swizzling logic:
// def get_tiled_id_triton(x, y, grid_size_x, grid_size_y, tile):
//     GROUP_SIZE_M = tile
//     pid = x + y * grid_size_x
//     # Number of programs in group
//     num_pid_in_group = GROUP_SIZE_M * grid_size_x
//     # Id of the group this program is in
//     group_id = pid // num_pid_in_group
//     # Row-id of the first program in the group
//     first_pid_m = group_id * GROUP_SIZE_M
//     # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`
//     group_size_m = min(grid_size_y - first_pid_m, GROUP_SIZE_M)
//     # *Within groups*, programs are ordered in a column-major order
//     # Row-id of the program in the *launch grid*
//     pid_m = first_pid_m + (pid % group_size_m)
//     # Col-id of the program in the *launch grid*
//     pid_n = (pid % num_pid_in_group) // group_size_m
// return pid_n, pid_m
static std::pair<Value, Value>
makeSwizzledIdsInTritonWay(Location loc, OpBuilder &b, Value x, Value y,
                           Value gridSizeX, Value gridSizeY,
                           unsigned swizzleTile) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value groupSizeM = b.create<arith::ConstantIndexOp>(loc, swizzleTile);
  // pid = x + y * grid_size_x
  Value pid = b.create<arith::AddIOp>(
      loc, x, b.create<arith::MulIOp>(loc, y, gridSizeX));

  // num_pid_in_group = GROUP_SIZE_M * grid_size_x
  Value numPidInGroup = b.create<arith::MulIOp>(loc, groupSizeM, gridSizeX);
  // group_id = pid / num_pid_in_group
  Value groupId = b.create<arith::DivUIOp>(loc, pid, numPidInGroup);
  // first_pid_m = group_id * GROUP_SIZE_M
  Value firstPidM = b.create<arith::MulIOp>(loc, groupId, groupSizeM);
  // group_size_m = min(grid_size_y - first_pid_m, GROUP_SIZE_M)
  Value gridSizeYMinusFirstPidM =
      b.create<arith::SubIOp>(loc, gridSizeY, firstPidM);
  Value groupSizeMFinal =
      b.create<arith::MinSIOp>(loc, gridSizeYMinusFirstPidM, groupSizeM);

  // pid_m = first_pid_m + (pid % group_size_m)
  Value pidModGroupSizeM = b.create<arith::RemUIOp>(loc, pid, groupSizeMFinal);
  Value pidM = b.create<arith::AddIOp>(loc, firstPidM, pidModGroupSizeM);

  // pid_n = (pid % num_pid_in_group) / group_size_m
  Value pidModNumPidInGroup = b.create<arith::RemUIOp>(loc, pid, numPidInGroup);
  Value pidN =
      b.create<arith::DivUIOp>(loc, pidModNumPidInGroup, groupSizeMFinal);

  return {pidN, pidM};
}

// Only support 2d grid.
static LogicalResult
reorderForallOpMappedTo2DBlock(scf::ForallOp forallOp unsigned swizzleLogTile) {
  unsigned swizzleTile = 1 << swizzleLogTile;
  OpBuilder b(forallOp);

  scf::ForallOp newforallOp = forallOp.clone();
  newforallOp.getBody()->clear();
  b.insert(newforallOp);
  b.setInsertionPointToStart(newforallOp.getBody());

  auto originLoops = forallOp.getInductionVars();
  auto gridSize = newforallOp.getUpperBound(b);
  auto loops = newforallOp.getInductionVars();
  auto mapping = newforallOp.getMappingAttr().getValue();

  Value workgroupIdX, workgroupIdY, workgroupCountX, workgroupCountY;
  if (mapping[0].cast<gpu::GPUBlockMappingAttr>().getMappingId() == 0) {
    workgroupIdX = loops[0];
    workgroupIdY = loops[1];
    workgroupCountX = gridSize[0];
    workgroupCountY = gridSize[1];
  } else {
    workgroupIdX = loops[1];
    workgroupIdY = loops[0];
    workgroupCountX = gridSize[1];
    workgroupCountY = gridSize[0];
  }

  auto [swizzledIdX, swizzledIdY] = makeSwizzledIdsInTritonWay(
      newforallOp.getLoc(), b, workgroupIdX, workgroupIdY, workgroupCountX,
      workgroupCountY, swizzleTile);

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
    func::FuncOp funcOp = getOperation();
    if (!hasGemmTileConfig(funcOp)) {
      return signalPassFailure();
    }

    auto forallOpOptional = getForallOpMappedTo2DBlock(funcOp);
    if (!forallOpOptional.has_value()) {
      return signalPassFailure();
    }
    scf::ForallOp forallOp = *forallOpOptional;

    if (failed(reorderForallOpMappedTo2DBlock(forallOp, swizzleLogTile))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUBlockSwizzlePass(int64_t swizzleLogTile) {
  return std::make_unique<GPUBlockSwizzlePass>(swizzleLogTile);
}
