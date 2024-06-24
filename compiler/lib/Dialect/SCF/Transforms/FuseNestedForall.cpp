//===- FuseNestedForall.cpp ------------------------------------ C++ --===//
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

#include "byteir/Dialect/SCF/Transforms/FuseNestedForall.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include <utility>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::scf;

namespace {

static bool checkMappingAttributeTypes(SmallVector<Attribute> mapping) {
  if (mapping.empty()) {
    return true;
  }

  bool hasBlockMapping = llvm::any_of(mapping, [](Attribute attr) {
    return isa<mlir::gpu::GPUBlockMappingAttr>(attr);
  });
  bool hasWarpgroupMapping = llvm::any_of(mapping, [](Attribute attr) {
    return isa<mlir::gpu::GPUWarpgroupMappingAttr>(attr);
  });
  bool hasWarpMapping = llvm::any_of(mapping, [](Attribute attr) {
    return isa<mlir::gpu::GPUWarpMappingAttr>(attr);
  });
  bool hasThreadMapping = llvm::any_of(mapping, [](Attribute attr) {
    return isa<mlir::gpu::GPUThreadMappingAttr>(attr);
  });
  int64_t countMappingTypes = 0;
  countMappingTypes += hasBlockMapping ? 1 : 0;
  countMappingTypes += hasWarpgroupMapping ? 1 : 0;
  countMappingTypes += hasWarpMapping ? 1 : 0;
  countMappingTypes += hasThreadMapping ? 1 : 0;
  if (countMappingTypes > 1) {
    return false;
  }

  llvm::DenseSet<Attribute> seen;
  for (Attribute map : mapping) {
    if (seen.contains(map)) {
      return false;
    }
    seen.insert(map);
  }

  auto isLinear = [](Attribute a) {
    return cast<DeviceMappingAttrInterface>(a).isLinearMapping();
  };

  if (llvm::any_of(mapping, isLinear) && !llvm::all_of(mapping, isLinear)) {
    return false;
  }

  return true;
}

bool isPerfectNestedForall(scf::ForallOp parentForall,
                           scf::ForallOp nestedForall) {
  Block &body = parentForall.getRegion().front();
  scf::InParallelOp parentReturnOp = parentForall.getTerminator();
  scf::InParallelOp nestedReturnOp = nestedForall.getTerminator();

  // InParallelOp has a single region with a single block
  if (!parentReturnOp.getRegion().front().empty() ||
      !nestedReturnOp.getRegion().front().empty())
    return false;

  Operation *lastOp = &(*std::prev(body.end(), 2));

  if (!llvm::isa<scf::ForallOp>(lastOp) ||
      lastOp != nestedForall.getOperation()) {
    return false;
  }

  SmallVector<OpFoldResult> mixedLb = nestedForall.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedUb = nestedForall.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStep = nestedForall.getMixedStep();

  auto isValueInParentForallBody =
      [&](const SmallVector<OpFoldResult> &config) -> bool {
    for (OpFoldResult ofr : config) {
      auto maybeCst = getConstantIntValue(ofr);
      if (!maybeCst.has_value()) {
        Value v = ofr.get<Value>();
        if (v.getParentBlock() == &body) {
          return true;
        }
      }
    }
    return false;
  };

  if (isValueInParentForallBody(mixedLb) ||
      isValueInParentForallBody(mixedUb) ||
      isValueInParentForallBody(mixedStep)) {
    return false;
  }

  bool parentHasMapping = parentForall.getMapping().has_value();
  bool nestedHasMapping = nestedForall.getMapping().has_value();
  if (nestedHasMapping != parentHasMapping) {
    return false;
  }

  if (!parentHasMapping && !nestedHasMapping) {
    return true;
  }

  auto mappingAttrs = llvm::to_vector(parentForall.getMappingAttr());
  mappingAttrs.append(nestedForall.getMappingAttr().begin(),
                      nestedForall.getMappingAttr().end());
  size_t numLoops = parentForall.getInductionVars().size() +
                    nestedForall.getInductionVars().size();
  if (numLoops != mappingAttrs.size() ||
      !checkMappingAttributeTypes(mappingAttrs)) {
    return false;
  }

  return true;
}

scf::ForallOp fuseNestedForallImpl(scf::ForallOp parentForall,
                                   scf::ForallOp nestedForall) {
  IRRewriter rewriter(parentForall.getContext());
  Location loc = parentForall.getLoc();

  auto outputs = llvm::to_vector(parentForall.getOutputs());
  auto nestOutpus = llvm::to_vector(nestedForall.getOutputs());
  outputs.append(nestOutpus.begin(), nestOutpus.end());

  SmallVector<OpFoldResult> mixedLb = parentForall.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedUb = parentForall.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStep = parentForall.getMixedStep();
  SmallVector<OpFoldResult> nestedLb = nestedForall.getMixedLowerBound();
  SmallVector<OpFoldResult> nestedUb = nestedForall.getMixedUpperBound();
  SmallVector<OpFoldResult> nestedStep = nestedForall.getMixedStep();

  mixedLb.append(nestedLb.begin(), nestedLb.end());
  mixedUb.append(nestedUb.begin(), nestedUb.end());
  mixedStep.append(nestedStep.begin(), nestedStep.end());

  SmallVector<Attribute> mappingAttrs;
  bool parentHasMapping = parentForall.getMapping().has_value();
  bool nestedHasMapping = nestedForall.getMapping().has_value();

  if (parentHasMapping) {
    mappingAttrs.append(parentForall.getMappingAttr().begin(),
                        parentForall.getMappingAttr().end());
  }
  if (nestedHasMapping) {
    mappingAttrs.append(nestedForall.getMappingAttr().begin(),
                        nestedForall.getMappingAttr().end());
  }

  rewriter.setInsertionPoint(parentForall);
  std::optional<ArrayAttr> maybeMapping = std::nullopt;
  if (mappingAttrs.size() > 0) {
    maybeMapping = rewriter.getArrayAttr(mappingAttrs);
  }
  auto newForallOp =
      rewriter.create<scf::ForallOp>(loc, mixedLb, mixedUb, mixedStep, outputs,
                                     maybeMapping);
  newForallOp.getTerminator()->erase();

  Block *parentForallLoopBody = parentForall.getBody();
  Block *newLoopBody = newForallOp.getBody();
  rewriter.mergeBlocks(parentForallLoopBody, newLoopBody,
                       newLoopBody->getArguments().take_front(
                           parentForallLoopBody->getNumArguments()));

  scf::ForallOp clonedNestedForallOp =
      cast<scf::ForallOp>(&(*std::prev(newLoopBody->end(), 2)));

  rewriter.setInsertionPoint(newForallOp.getTerminator());
  IRMapping bvm;
  for (auto [oldIv, newIv] :
       llvm::zip_equal(clonedNestedForallOp.getInductionVars(),
                       newLoopBody->getArguments().take_back(
                           clonedNestedForallOp.getInductionVars().size()))) {
    bvm.map(oldIv, newIv);
  }

  for (Operation &op : clonedNestedForallOp.getBody()->without_terminator())
    rewriter.clone(op, bvm);

  clonedNestedForallOp->erase();
  parentForall->erase();

  return newForallOp;
}

struct FuseNestedForallPass
    : public FuseNestedForallBase<FuseNestedForallPass> {
  FuseNestedForallPass(llvm::StringRef anchor) : FuseNestedForallBase() {
    anchorTag = anchor.str();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // skip non-anchored
    if (!anchorTag.empty() && !funcOp->hasAttr(anchorTag)) {
      return;
    }

    llvm::DenseMap<scf::ForallOp, SmallVector<scf::ForallOp>> fuseCluster;
    funcOp->walk([&](scf::ForallOp curForallOp) {
      fuseCluster[curForallOp] = {curForallOp};
    });

    funcOp->walk([&](scf::ForallOp curForallOp) {
      if (auto parentForallOp = curForallOp->getParentOfType<scf::ForallOp>()) {
        if (isPerfectNestedForall(parentForallOp, curForallOp)) {
          fuseCluster[parentForallOp].append(fuseCluster[curForallOp].begin(),
                                             fuseCluster[curForallOp].end());
          fuseCluster.erase(curForallOp);
        }
      }
    });

    for (const auto &cluster : fuseCluster) {
      if (cluster.second.size() < 2) {
        continue;
      }
      // from inside out
      auto loops = llvm::to_vector(llvm::reverse(cluster.second));
      scf::ForallOp nestedForall = loops[0];
      for (size_t i = 1; i < loops.size(); ++i) {
        nestedForall = fuseNestedForallImpl(loops[i], nestedForall);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createFuseNestedForallPass(llvm::StringRef anchor) {
  return std::make_unique<FuseNestedForallPass>(anchor);
}
