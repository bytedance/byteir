//===- GPUPackSharedMemory.cpp --------------------------*--- C++-*-===//
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
// Some code comes from
// compiler/src/iree/compiler/Codegen/LLVMGPU/GPUPackSharedMemory.cpp of
// IREE project.
// Original license:
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "byteir/Dialect/GPU/Transforms/GPUPackSharedMemory.h"
#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "PassDetail.h"

#define DEBUG_TYPE "gpu-pack-shared-memory"

using namespace mlir;

namespace {
// 
static int64_t getAllocSize(Operation *op, DataLayout &dataLayout) {
  auto allocOp = cast<memref::AllocaOp>(op);
  int64_t numElements = allocOp.getType().getNumElements();
  return (dataLayout.getTypeSizeInBits(allocOp.getType().getElementType()) *
          numElements) /
         8;
}

// help function of packing
void analyseAllocsForPacking(func::FuncOp funcOp, ArrayRef<Operation *> allocs,
                             SmallVector<AliasGroup> &aliasGroups) {
  // Represent of a group of allocations with overlapping liverange and the
  // liveness of the overall group.
  struct AllocGroup {
    SmallVector<Operation *> allocs;
    // Keep track of every operation where any of the alloc in the group is
    // live.
    // Liveness is represent as a set of Operations where the alloc is alive.
    // To make it merge liveranges and check if a given Operation interfers
    // with the liverange we store it as a DesneSet.
    llvm::DenseSet<Operation *> liveness;
  };
  Liveness liveness(funcOp);
  SmallVector<AllocGroup> groups;
  for (Operation *alloc : allocs) {
    SmallVector<size_t> aliasGroups;
    for (size_t i : llvm::seq<size_t>(0, groups.size())) {
      AllocGroup &group = groups[i];
      for (Operation *user : alloc->getUsers()) {
        // Skip the whole analysis if any user is a subview.
        // TODO: This could be extended if needed by recursively merging
        // liveness.
        if (isa<memref::SubViewOp>(user))
          return;
        if (group.liveness.count(user)) {
          aliasGroups.push_back(i);
          break;
        }
      }
    }
    if (aliasGroups.empty()) {
      // If we didn't find any alias group create a new one.
      AllocGroup &newGroup = groups.emplace_back();
      newGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo =
          liveness.resolveLiveness(alloc->getResult(0));
      newGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
    } else {
      // Merge the alloc into the first alias group it interfers with.
      AllocGroup &mergeGroup = groups[aliasGroups[0]];
      mergeGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo =
          liveness.resolveLiveness(alloc->getResult(0));
      mergeGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
      // Then merge all the other alias groups into the first group.
      for (size_t i = 1, e = aliasGroups.size(); i < e; i++) {
        AllocGroup &group = groups[aliasGroups[i]];
        mergeGroup.allocs.insert(mergeGroup.allocs.end(), group.allocs.begin(),
                                 group.allocs.end());
        mergeGroup.liveness.insert(group.liveness.begin(),
                                   group.liveness.end());
        // For simplicity we leave the group empty and don't remove it.
        group.allocs.clear();
        group.liveness.clear();
      }
    }
  }

  LLVM_DEBUG({
    for (size_t i = 0; i < groups.size(); i++) {
      llvm::dbgs() << "Alias group " << i << ":\n";
      for (Operation *op : groups[i].allocs)
        op->dump();
    }
  });

  for (size_t i = 0; i < groups.size(); i++) {
    if (groups[i].allocs.empty())
      continue;
    aliasGroups.push_back(std::move(groups[i].allocs));
  }
}

void packAllocs(OpBuilder &builder, scf::ForallOp forallOp,
                ArrayRef<AliasGroup> aliasGroups) {
  if (aliasGroups.empty())
    return;
  DataLayout dataLayout = DataLayout::closest(funcOp);
  scf::ForallOp loopParent =
      aliasGroups[0][0]->getParentOfType<scf::ForallOp>();
  builder.setInsertionPointToStart(forallOp.getBody());
  int64_t maxAlloc = 0;
  for (size_t i = 0; i < aliasGroups.size(); i++) {
    int64_t allocSize = 0;
    for (Operation *alloc : aliasGroups[i]) {
      allocSize += getAllocSize(alloc, dataLayout);
    }
    maxAlloc = std::max(maxAlloc, allocSize);
  }
  Attribute memorySpace =
      llvm::cast<MemRefType>(aliasGroups[0][0]->getResultTypes()[0])
          .getMemorySpace();
  // Alloc according to the max alloc size among groups.
  MemRefType allocType = MemRefType::get({maxAlloc}, builder.getI8Type(),
                                         AffineMap(), memorySpace);
  Value packedAlloc =
      builder.create<memref::AllocaOp>(funcOp.getLoc(), allocType);
  for (size_t i = 0; i < aliasGroups.size(); i++) {
    int64_t offset = 0;
    for (Operation *alloc : aliasGroups[i]) {
      Location loc = alloc->getLoc();
      builder.setInsertionPoint(alloc);
      Value offsetValue = builder.create<arith::ConstantIndexOp>(loc, offset);
      Value newAlloc = builder.create<memref::ViewOp>(
          packedAlloc.getLoc(), alloc->getResultTypes()[0], packedAlloc,
          offsetValue, ArrayRef<Value>({}));
      offset += getAllocSize(alloc, dataLayout);
      alloc->replaceAllUsesWith(ArrayRef<Value>({newAlloc}));
      alloc->erase();
    }
  }
}

void sinkOpsInCFG(const SmallVector<Operation *> &allocs,
                  DominanceInfo &dominators) {
  for (Operation *sinkOp : allocs) {
    Block *dom = nullptr;
    for (Operation *user : sinkOp->getUsers()) {
      if (!dom) {
        dom = user->getBlock();
        // Find the block in the same region.
        while (dom->getParent() != sinkOp->getParentRegion()) {
          dom = dom->getParentOp()->getBlock();
        }
        continue;
      }
      dom = dominators.findNearestCommonDominator(dom, user->getBlock());
    }
    llvm::SmallDenseSet<Operation *> users;
    for (Operation *user : sinkOp->getUsers()) {
      while (user->getParentRegion() != sinkOp->getParentRegion()) {
        user = user->getParentOp();
      }
      users.insert(user);
    }
    Operation *firstUse = dom->getTerminator();
    for (Operation &op : dom->getOperations()) {
      if (users.count(&op)) {
        firstUse = &op;
        break;
      }
    }
    sinkOp->moveBefore(firstUse);
  }
}

void packSharedMemoryAlloc(func::FuncOp funcOp) {
  DominanceInfo dominators(funcOp);
  SmallVector<Operation *> allocs;
  funcOp.walk([&](memref::AllocaOp alloca) {
    if (hasSharedMemoryAddressSpace(alloca.getType())) {
      allocs.push_back(alloca);
    }
  });
  // First sink the alloc as low as possible in the CFG.
  sinkOpsInCFG(allocs, dominators);
  SmallVector<AliasGroup> aliasGroups;
  analyseAllocsForPacking(funcOp, allocs, aliasGroups);
  // If there is 1 or less alias group there is nothing to do.
  if (aliasGroups.size() <= 1)
    return;

  // Pack all the allocations into one i8 alloc.
  // We may need to add extra barriers to make sure we are done writting or
  // reading from the previous alias group before starting a new one.
  // TODO(Yangxinyu): analysis this
  int sz = aliasGroups.size();
  // insert barrier at last aliasGroup
  for (Operation *alloc : aliasGroups[0]) {
    addBarrier(funcOp, alloc, aliasGroups[0]);
  }
  // for (size_t i = 0; i < aliasGroups.size(); i++) {
  //   for (Operation *alloc : aliasGroups[i]) {
  //     addBarrier(funcOp, alloc, aliasGroups[i]);
  //   }
  // }

  OpBuilder builder(funcOp.getContext());
  packAllocs(builder, funcOp, aliasGroups);
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createPackSharedMemoryPass() {
  return std::make_unique<PackSharedMemoryPass>();
}