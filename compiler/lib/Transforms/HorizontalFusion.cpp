//===- HorizontalFusion.cpp ----------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/HorizontalFusion.h"

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <numeric>

#include "PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

constexpr StringRef kernelTypeNameAttr = "__byteir_forall_type_name";
constexpr StringRef kernelFuncNameAttr = "__byteir_forall_kernel_name";

void setForallTagAndName(ModuleOp &m) {
  static constexpr StringRef elementwiseAttrName =
      getByteIRElementwiseFusionAttrName();
  static constexpr StringRef reductionAttrName =
      getByteIRReductionFusionAttrName();

  for (auto funcOp : m.getOps<func::FuncOp>()) {
    auto funcName = funcOp.getName();
    StringRef kernelType;
    if (funcOp->hasAttr(elementwiseAttrName)) {
      kernelType = elementwiseAttrName;
    } else if (funcOp->hasAttr(reductionAttrName)) {
      kernelType = reductionAttrName;
    } else
      continue;

    mlir::OpBuilder opBuilder(funcOp);
    for (auto forallOp : funcOp.getOps<scf::ForallOp>()) {
      forallOp->setAttr(kernelTypeNameAttr,
                        opBuilder.getStringAttr(kernelType));
      forallOp->setAttr(kernelFuncNameAttr, opBuilder.getStringAttr(funcName));
    }
  }
}

inline bool isByreEntry(func::FuncOp &funcOp) {
  return funcOp->hasAttr(getAttrPlaceholderName(
      byre::ByreDialect::getEntryPointFunctionAttrName()));
}

bool isDefInParent(Value val, Operation *parent) {
  auto defOp = val.getDefiningOp();
  while (defOp) {
    if (defOp->getParentOp() == parent)
      return true;
    defOp = defOp->getParentOp();
  }
  return false;
}

void moveForwardAlloc(ModuleOp &m) {
  for (auto funcOp : m.getOps<func::FuncOp>()) {
    if (!isByreEntry(funcOp))
      continue;
    mlir::OpBuilder b(funcOp);
    b.setInsertionPointToStart(&(funcOp.getBody().front()));
    Block::iterator insertionPoint = b.getInsertionPoint();
    for (auto alloc :
         llvm::make_early_inc_range(funcOp.getOps<memref::AllocOp>())) {
      alloc->moveAfter(&(funcOp.getBody().front()), insertionPoint);
    }
  }
}

template <typename attrTy> inline bool hasMappingAttr(scf::ForallOp forall) {
  auto mapping = forall.getMapping();
  bool hasAttr =
      mapping.has_value() && llvm::any_of(mapping.value(), [&](Attribute attr) {
        return isa<attrTy>(attr);
      });
  return hasAttr;
};

int64_t getForallLoopSize(scf::ForallOp &forall) {
  auto ubs = forall.getStaticUpperBound();
  auto lbs = forall.getStaticLowerBound();
  auto steps = forall.getStaticStep();
  int64_t loopSize = 1;
  for (auto &&[ub, lb, s] : llvm::zip(ubs, lbs, steps)) {
    loopSize *= (ub - lb) / s;
  }
  return loopSize;
}

int64_t extractForallBlockSize(Operation *op) {
  int64_t blockSize = 0;
  if (auto forall = dyn_cast_if_present<scf::ForallOp>(op)) {
    // FIXME. assume only one forall in outter forall body.
    forall->walk([&](scf::ForallOp innerForall) {
      auto mapping = innerForall.getMapping();
      if (!mapping.has_value())
        return WalkResult::advance();
      bool hasThreadMapping = hasMappingAttr<gpu::GPUThreadMappingAttr>(forall);
      if (hasThreadMapping) {
        blockSize = getForallLoopSize(innerForall);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
  return blockSize;
}

int64_t extractForallGridSize(Operation *op) {
  int64_t gridSize = 0;
  if (auto forall = dyn_cast_if_present<scf::ForallOp>(op)) {
    if (hasMappingAttr<gpu::GPUBlockMappingAttr>(forall)) {
      gridSize = getForallLoopSize(forall);
    }
  }
  return gridSize;
}

int64_t extractForallInstrCount(Operation *op) {
  int instrCount = 0;
  if (auto forall = dyn_cast_if_present<scf::ForallOp>(op)) {
    // TODO skip viewlike ops.
    instrCount = std::distance(forall.getOps().begin(), forall.getOps().end());
    for (auto innerForall : forall.getOps<scf::ForallOp>()) {
      innerForall->walk([&](Operation *instr) { instrCount++; });
    }
  }
  return instrCount;
}

// HorizontalFusionPass
using HFusionPattern = llvm::SmallVector<Operation *, 8>;
using HFusionPlan = llvm::SmallVector<HFusionPattern, 8>;

// TODO(chhuang)
constexpr int64_t kMAX_INSTR_COUNT = 1024;
constexpr int64_t kMAX_BLOCK_SIZE = 8192 * 8192;
constexpr int64_t kMAX_GRID_SIZE = 1024;

struct HorizontalFusionPass
    : public HorizontalFusionBase<HorizontalFusionPass> {
  explicit HorizontalFusionPass()
      : HorizontalFusionBase<HorizontalFusionPass>::HorizontalFusionBase() {}

  void runOnOperation() override;
  void getCandidates(ModuleOp &m, SmallVector<Operation *> &candidates);
  void makeHorizontalFusionPlan(SmallVector<Operation *> &, HFusionPlan &);
  void doHorizontalFusion(HFusionPlan &);
  bool isFusibleAndBenefit(scf::ForallOp pre, scf::ForallOp cur);
  void collectWRMemref(scf::ForallOp forallOp, llvm::SetVector<Value> &w,
                       llvm::SetVector<Value> &r);
  void collectDirectOperands(scf::ForallOp forallOp,
                             llvm::SetVector<Operation *> &vals);
  void collectUsePointInBlock(Block *block, llvm::SetVector<Value> &vals,
                              llvm::SetVector<Operation *> &usePoints);
}; // HorizontalFusionPass

void HorizontalFusionPass::getCandidates(ModuleOp &m,
                                         SmallVector<Operation *> &candidates) {
  auto hasGPUMapping = [](scf::ForallOp forall) {
    bool hasBlockMapping = hasMappingAttr<gpu::GPUBlockMappingAttr>(forall);
    if (!hasBlockMapping)
      return false;
    bool hasThreadMapping = !forall.getOps<scf::ForallOp>().empty();
    for (auto innerForall : forall.getOps<scf::ForallOp>()) {
      hasThreadMapping &=
          hasMappingAttr<gpu::GPUThreadMappingAttr>(innerForall);
    }
    return hasThreadMapping;
  };

  // Fuse too much instrs may lead to register spill and use too much resource.
  auto checkInstrCount = [](scf::ForallOp forall, const int64_t max_instr) {
    return extractForallInstrCount(forall.getOperation()) < max_instr;
  };

  // It's not benefit to fuse kernel with large grid size, which already has
  // enough blocks to occupy GPU SMs.
  auto checkGridSize = [](scf::ForallOp forall, const int64_t max_grid) {
    return extractForallGridSize(forall) < max_grid;
  };

  auto checkBlockSize = [](scf::ForallOp forall, const int64_t max_block) {
    return extractForallBlockSize(forall) < max_block;
  };

  auto checkAll = [&](scf::ForallOp forall) {
    bool check = true;
    check &= hasGPUMapping(forall);
    check &= checkInstrCount(forall, kMAX_INSTR_COUNT);
    check &= checkGridSize(forall, kMAX_GRID_SIZE);
    check &= checkBlockSize(forall, kMAX_BLOCK_SIZE);

    return check;
  };

  for (auto funcOp : m.getOps<func::FuncOp>()) {
    if (!isByreEntry(funcOp))
      continue;

    for (auto forallOp : funcOp.getOps<scf::ForallOp>()) {
      // TODO(chhuang) (1) check instrs nums; (2) skip large shape;
      if (forallOp->hasAttr(kernelTypeNameAttr) &&
          forallOp->getAttr(kernelTypeNameAttr).cast<StringAttr>().getValue() ==
              getByteIRElementwiseFusionAttrName() &&
          checkAll(forallOp)) {
        candidates.push_back(forallOp);
      }
    }
  }
}

// traverse from top to down, greedy check whether fuseiable
void HorizontalFusionPass::makeHorizontalFusionPlan(
    SmallVector<Operation *> &candidates, HFusionPlan &plan) {
  Operation *head = nullptr;
  HFusionPattern *pattern = nullptr;
  for (auto cur : candidates) {
    if (head && isFusibleAndBenefit(dyn_cast<scf::ForallOp>(head),
                                    dyn_cast<scf::ForallOp>(cur))) {
      pattern->push_back(cur);
      continue;
    }
    head = cur;
    HFusionPattern newPattern;
    plan.push_back(newPattern);
    pattern = &(plan.back());
    pattern->push_back(head);
  }
}

void HorizontalFusionPass::doHorizontalFusion(HFusionPlan &plan) {
  OpBuilder builder(getOperation());
  for (auto &&pattern : plan) {
    if (pattern.size() < 2)
      continue;
    // Sort forall with shape and insts count. So the same blocksize will be
    // placed adjacent each other and fuse together computations of similar
    // sizes.
    std::sort(pattern.begin(), pattern.end(), [&](Operation *a, Operation *b) {
      auto aBlockSize = extractForallBlockSize(a);
      auto bBlockSize = extractForallBlockSize(b);
      if (aBlockSize != bBlockSize)
        return aBlockSize < bBlockSize;
      return extractForallInstrCount(a) < extractForallInstrCount(b);
    });

    // merge
    auto root = cast<scf::ForallOp>(pattern.front());
    SmallVector<int64_t, 3> blockNums;
    for (auto op : pattern) {
      auto forall = cast<scf::ForallOp>(op);
      blockNums.push_back(forall.getStaticUpperBound().front());
    }
    // TODO should we align blockNum to multiple 32 to reduce divergence
    int64_t allBlockNums = std::accumulate(blockNums.begin(), blockNums.end(),
                                           0, std::plus<int64_t>());

    auto front = cast<scf::ForallOp>(pattern.front());
    auto loc = front.getLoc();
    builder.setInsertionPoint(front);
    SmallVector<Value> bounds;
    Value tempBound = builder.create<arith::ConstantIndexOp>(loc, 0);
    bounds.push_back(tempBound);
    for (auto num : blockNums) {
      Value n = builder.create<arith::ConstantIndexOp>(loc, num);
      tempBound = builder.create<arith::AddIOp>(loc, tempBound, n);
      bounds.push_back(tempBound);
    }

    auto cstIZero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto cstIOne = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto cstNums = builder.create<arith::ConstantIndexOp>(loc, allBlockNums);
    // FIXME not hack lb and step here
    auto lb = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto step = builder.create<arith::ConstantIndexOp>(loc, 1);

    // create grid level forall
    auto hFuseForall =
        builder.create<scf::ForallOp>(loc, /*lb*/ ArrayRef<OpFoldResult>({lb}),
                                      /*ub*/ ArrayRef<OpFoldResult>({cstNums}),
                                      /*step*/ ArrayRef<OpFoldResult>({step}),
                                      ValueRange(), front.getMapping());
    builder.setInsertionPointToStart(hFuseForall.getBody());
    auto blockId = hFuseForall.getBody()->getArgument(0);

    // create condition br one by one
    Value switchValue = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (int64_t i = 0; i < blockNums.size(); ++i) {
      auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                               blockId, bounds[i + 1]);
      auto selVal =
          builder.create<arith::SelectOp>(loc, cmp, cstIOne, cstIZero);
      switchValue = builder.create<arith::AddIOp>(loc, switchValue, selVal);
    }
    SmallVector<int64_t> cases =
        llvm::to_vector(llvm::seq<int64_t>(0, blockNums.size()));
    auto switchOp = builder.create<scf::IndexSwitchOp>(
        loc, /*resultTypes*/ TypeRange{}, switchValue, cases, cases.size());

    // default region
    {
      Block &defaultBlock = switchOp.getDefaultRegion().emplaceBlock();
      builder.setInsertionPointToStart(&defaultBlock);
      builder.create<scf::YieldOp>(loc);
    }

    // case region
    for (int64_t i = 0; i < blockNums.size(); ++i) {
      auto orgForall = cast<scf::ForallOp>(pattern[i]);
      Block &caseBlock = switchOp.getCaseRegions()[i].emplaceBlock();
      builder.setInsertionPointToStart(&caseBlock);
      auto orgId = builder.create<arith::SubIOp>(loc, blockId, bounds[i]);
      Block::iterator insertionPoint = builder.getInsertionPoint();
      replaceAllUsesInRegionWith(orgForall.getBody()->getArgument(0), orgId,
                                 orgForall.getRegion());
      caseBlock.getOperations().splice(insertionPoint,
                                       orgForall.getBody()->getOperations());
      caseBlock.back().erase();
      builder.create<scf::YieldOp>(loc);
      orgForall.erase();
    }
  }
}

bool HorizontalFusionPass::isFusibleAndBenefit(scf::ForallOp pre,
                                               scf::ForallOp cur) {
  // TODO check whether benefit

  auto same_shape = [](ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
    if (a.size() != b.size())
      return false;
    return llvm::all_of((llvm::zip(a, b)), [](std::tuple<int64_t, int64_t> s) {
      return std::get<0>(s) == std::get<1>(s);
    });
  };

  // check fusiable
  llvm::SetVector<Value> curWriteVals;
  llvm::SetVector<Value> curReadVals;
  // include alias and collect all uses.
  collectWRMemref(cur, curWriteVals, curReadVals);

  llvm::SetVector<Operation *> directOperands;
  // collectDirectOperands(cur, directOperands);

  llvm::SetVector<Operation *> usePoints;
  collectUsePointInBlock(cur->getParentOp()->getBlock(), curWriteVals,
                         usePoints);
  collectUsePointInBlock(cur->getParentOp()->getBlock(), curReadVals,
                         usePoints);

  auto &domInfo = getAnalysis<DominanceInfo>();

  auto checkUsePointDominace = [&](Operation *op) {
    // just skip checking viewlike ops
    if (isa_and_nonnull<ViewLikeOpInterface>(op))
      return true;
    return domInfo.properlyDominates(op, pre) || domInfo.dominates(cur, op);
  };

  auto checkDirectOpdDominace = [&](Operation *op) {
    return domInfo.properlyDominates(op, pre);
  };

  bool fusiable = llvm::all_of(usePoints, checkUsePointDominace);
  fusiable &= llvm::all_of(directOperands, checkDirectOpdDominace);

  return fusiable;
}

void HorizontalFusionPass::collectWRMemref(scf::ForallOp forallOp,
                                           llvm::SetVector<Value> &w,
                                           llvm::SetVector<Value> &r) {
  auto collect = [](Value memref, llvm::SetVector<Value> &chunk) {
    Value root = memref;
    while (true) {
      if (auto defOp =
              dyn_cast_if_present<ViewLikeOpInterface>(root.getDefiningOp())) {
        root = defOp->getOperand(0);
        continue;
      }
      break;
    }
    SmallVector<Value> worklist;
    chunk.insert(root);
    worklist.push_back(root);
    while (!worklist.empty()) {
      auto val = worklist.pop_back_val();
      for (auto user : val.getUsers()) {
        if (auto viewlike = dyn_cast_if_present<ViewLikeOpInterface>(user)) {
          for (auto res : viewlike->getResults()) {
            worklist.push_back(res);
            chunk.insert(res);
          }
        }
      }
    }
  };

  forallOp->walk([&](memref::LoadOp load) {
    auto memref = load.getMemref();
    collect(memref, r);
  });
  forallOp->walk([&](memref::StoreOp store) {
    auto memref = store.getMemref();
    collect(memref, w);
  });

  forallOp->walk([&](Operation *op) {
    if (!isa<memref::MemRefDialect>(op->getDialect()))
      return WalkResult::advance();
    for (auto opd : op->getOperands()) {
      if (isDefInParent(opd, forallOp.getOperation()))
        continue;
      bool readEffect = false;
      if (auto opInter = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (opInter.getEffectOnValue<MemoryEffects::Read>(opd).has_value()) {
          readEffect = true;
        }
      }
      if (readEffect)
        collect(opd, r);
      else
        collect(opd, w);
    }
    return WalkResult::advance();
  });
}

void HorizontalFusionPass::collectDirectOperands(
    scf::ForallOp forallOp, llvm::SetVector<Operation *> &vals) {
  auto collect = [](Value memref, llvm::SetVector<Operation *> &opds) {
    if (auto def = memref.getDefiningOp())
      opds.insert(def);
  };

  forallOp->walk([&](Operation *op) {
    if (!isa<memref::MemRefDialect>(op->getDialect()))
      return WalkResult::advance();
    for (auto opd : op->getOperands()) {
      if (isDefInParent(opd, forallOp.getOperation()))
        continue;
      if (!isa<MemRefType>(opd.getType()))
        continue;
      collect(opd, vals);
    }
    return WalkResult::advance();
  });
  // forallOp->walk([&](memref::LoadOp load) {
  //  auto memref = load.getMemref();
  //  collect(memref, vals);
  //});
  // forallOp->walk([&](memref::StoreOp store) {
  //  auto memref = store.getMemref();
  //  collect(memref, vals);
  //});
}

void HorizontalFusionPass::collectUsePointInBlock(
    Block *block, llvm::SetVector<Value> &vals,
    llvm::SetVector<Operation *> &usePoints) {
  for (auto val : vals) {
    for (auto user : val.getUsers()) {
      Operation *inBlockUser = user;
      while (inBlockUser->getParentOp() &&
             inBlockUser->getParentOp()->getBlock() != block) {
        inBlockUser = inBlockUser->getParentOp();
      }
      if (inBlockUser->getParentOp()) {
        usePoints.insert(inBlockUser);
      }
    }
  }
}

void HorizontalFusionPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  auto &domInfo = getAnalysis<DominanceInfo>();

  /// stage 1. inline all fused function back to entry function
  {
    setForallTagAndName(moduleOp);

    OpPassManager pm(moduleOp.getOperationName());
    pm.addPass(createInlinerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (mlir::failed(runPipeline(pm, moduleOp))) {
      signalPassFailure();
    }
  }
  /// stage 2. make fusion planing
  //  move forward alloc.
  moveForwardAlloc(moduleOp);
  // TODO move forward viewlike ops.

  SmallVector<Operation *> candidateForallOps;
  getCandidates(moduleOp, candidateForallOps);
  HFusionPlan horiFusionPlan;
  makeHorizontalFusionPlan(candidateForallOps, horiFusionPlan);

  /// stage 3. do horizontal fusion
  doHorizontalFusion(horiFusionPlan);

  /// postprocess
  // TODO lazy alloc
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createHorizontalFusionPass() {
  return std::make_unique<HorizontalFusionPass>();
}
