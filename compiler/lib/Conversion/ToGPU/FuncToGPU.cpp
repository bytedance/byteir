//===- FuncToGPU.cpp ---------------------------------------------- C++ -*-===//
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

#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToGPU/Utils.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/PipelineUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include <utility> // pair

#include "../PassDetail.h"

#define DEBUG_TYPE "func-to-gpu"

// TODO: configurable coarsen factor
#define COARSEN_FACTOR 1

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::gpu;

namespace {
constexpr int64_t kGridTileNumThreshold = 64;
constexpr int64_t kNumWave = 128;
constexpr int64_t kWarpSize = 32;

static void creaetGuardedSIMT(OpBuilder &b, Value id, Value bound,
                              LoopLikeOpInterface looplike, bool coarsen) {
  b.setInsertionPoint(looplike);

  if (coarsen) {
    auto newIV = createIndexValue(b, looplike, id);
    setLoopLowerBound(b, looplike, newIV);
    multiplyLoopStep(b, looplike, bound);

    // remove attrs
    looplike->removeAttr(getLoopToSIMTAttrName());
    looplike->removeAttr(getCoarsenSIMTAttrName());

    return;
  }

  IRMapping bvm;
  // newIV = lb + idx * step
  auto newIV = createIndexValue(b, looplike, id);
  auto oldIV = getInductionVar(looplike);
  bvm.map(oldIV, newIV);

  auto guardedBlock = createGuardedBranch(b, newIV, looplike);
  if (guardedBlock == nullptr)
    return;

  b.setInsertionPointToStart(guardedBlock);
  assert(llvm::hasSingleElement(looplike.getLoopRegions()));
  for (auto &block : *looplike.getLoopRegions().front()) {
    for (auto &op : block.without_terminator()) {
      b.clone(op, bvm);
    }
  }
  looplike.erase();
}

static void creaetGuardedSIMT(OpBuilder &b, func::FuncOp func,
                              LoopLikeOpInterface looplike,
                              GPUIndexType indexType, gpu::Dimension dim,
                              bool coarsen) {

  auto loc = looplike.getLoc();
  b.setInsertionPointToStart(&func.getBody().front());
  Value idx;
  Value bound;
  if (indexType == GPUIndexType::linear_id) {
    // idx = thread_id + block_id * block_dim
    auto bix = b.create<gpu::BlockIdOp>(loc, dim);
    auto bdim = b.create<gpu::BlockDimOp>(loc, dim);
    auto tix = b.create<gpu::ThreadIdOp>(loc, dim);
    idx = createLinearIndexValue(b, tix, bix, bdim);
    auto gdim = b.create<gpu::GridDimOp>(loc, dim);
    bound = b.create<arith::MulIOp>(loc, bdim, gdim);
  } else if (indexType == GPUIndexType::thread_id) {
    // idx = thread_id
    idx = b.create<gpu::ThreadIdOp>(loc, dim);
    bound = b.create<gpu::BlockDimOp>(loc, dim);
  } else if (indexType == GPUIndexType::block_id) {
    // idx = block_id
    idx = b.create<gpu::BlockIdOp>(loc, dim);
    bound = b.create<gpu::GridDimOp>(loc, dim);
  }

  creaetGuardedSIMT(b, idx, bound, looplike, coarsen);
}

static void convertLoopToSIMT(OpBuilder &b, func::FuncOp func,
                              LoopLikeOpInterface looplike) {
  auto strAttr = looplike->getAttrOfType<StringAttr>(getLoopToSIMTAttrName());
  auto coarsen = looplike->hasAttrOfType<UnitAttr>(getCoarsenSIMTAttrName());

  // default values
  GPUIndexType gpuIdxT = GPUIndexType::linear_id;
  gpu::Dimension dim = gpu::Dimension::x;

  if (strAttr.getValue() == getLinearIdXName()) {
    gpuIdxT = GPUIndexType::linear_id;
    dim = gpu::Dimension::x;
  } else if (strAttr.getValue() == getLinearIdYName()) {
    gpuIdxT = GPUIndexType::linear_id;
    dim = gpu::Dimension::y;
  } else if (strAttr.getValue() == getLinearIdZName()) {
    gpuIdxT = GPUIndexType::linear_id;
    dim = gpu::Dimension::z;
  } else if (strAttr.getValue() == getThreadIdXName()) {
    gpuIdxT = GPUIndexType::thread_id;
    dim = gpu::Dimension::x;
  } else if (strAttr.getValue() == getThreadIdYName()) {
    gpuIdxT = GPUIndexType::thread_id;
    dim = gpu::Dimension::y;
  } else if (strAttr.getValue() == getThreadIdZName()) {
    gpuIdxT = GPUIndexType::thread_id;
    dim = gpu::Dimension::z;
  } else if (strAttr.getValue() == getBlockIdXName()) {
    gpuIdxT = GPUIndexType::block_id;
    dim = gpu::Dimension::x;
  } else if (strAttr.getValue() == getBlockIdYName()) {
    gpuIdxT = GPUIndexType::block_id;
    dim = gpu::Dimension::y;
  } else if (strAttr.getValue() == getBlockIdZName()) {
    gpuIdxT = GPUIndexType::block_id;
    dim = gpu::Dimension::z;
  }

  creaetGuardedSIMT(b, func, looplike, gpuIdxT, dim, coarsen);
}

static void rewriteFuncImpl(OpBuilder &builder, func::FuncOp func) {
  SmallVector<LoopLikeOpInterface> loops;

  // collect loops from inner to outer
  func.walk([&](LoopLikeOpInterface loopLike) {
    if (loopLike->hasAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
      loops.push_back(loopLike);
    }
  });

  for (auto loop : loops) {
    convertLoopToSIMT(builder, func, loop);
  }
}

static std::pair<KernelDim3, KernelDim3> createBlockAndGrid(OpBuilder &b,
                                                            func::FuncOp func) {

  auto arrayAttr = func->getAttrOfType<ArrayAttr>(getToGPUAttrName());
  auto loc = func.getLoc();

  auto bx = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[0].cast<IntegerAttr>().getInt());
  auto by = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[1].cast<IntegerAttr>().getInt());
  auto bz = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[2].cast<IntegerAttr>().getInt());
  auto gx = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[3].cast<IntegerAttr>().getInt());
  auto gy = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[4].cast<IntegerAttr>().getInt());
  auto gz = b.create<arith::ConstantIndexOp>(
      loc, arrayAttr[5].cast<IntegerAttr>().getInt());

  KernelDim3 block{bx, by, bz};
  KernelDim3 grid{gx, gy, gz};

  return {block, grid};
}

static bool isGlobalAllocAlias(Operation &op) {
  if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::ReshapeOp>(
          op)) {
    return true;
    auto defOp = op.getOperand(0).getDefiningOp();
    if (defOp == nullptr)
      return true;
    return isGlobalAllocAlias(*defOp);
  }
  return isGPUGlobalAlloc(op);
};

static bool isHoistUpOp(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp, memref::CollapseShapeOp,
             memref::DimOp, memref::ExpandShapeOp, memref::ReshapeOp>(op);
}

static gpu::LaunchFuncOp rewriteToGPULaunchFuncImpl(OpBuilder &builder,
                                                    func::FuncOp func,
                                                    ArrayRef<Value> args,
                                                    gpu::GPUFuncOp gpuFunc) {
  // Rewrite Orignal function
  Region &funcBody = func.getBody();
  Block &funcEntryBlock = funcBody.front();
  SmallVector<Operation *> eraseOps;
  for (auto &op : funcEntryBlock.without_terminator()) {
    // FIXME: this might be buggy
    if (!isGlobalAllocAlias(op)) {
      eraseOps.push_back(&op);
    }
  }

  // TODO add
  for (auto it = eraseOps.rbegin(); it != eraseOps.rend(); ++it) {
    (*it)->erase();
  }

  builder.setInsertionPoint(funcEntryBlock.getTerminator());
  auto blockAndGrid = createBlockAndGrid(builder, func);
  auto launch = builder.create<gpu::LaunchFuncOp>(
      func.getLoc(), gpuFunc, blockAndGrid.second, blockAndGrid.first,
      /*dynamicSharedMemorySize=*/nullptr, args);
  return launch;
}

int64_t estimateGridSize(LoopLikeOpInterface loopLike, int64_t currGs,
                         int64_t stepMultiplier) {
  auto maybeTripCnt = getConstantTripCount(loopLike, stepMultiplier);

  if (maybeTripCnt.has_value() &&
      (*maybeTripCnt > static_cast<uint64_t>(currGs))) {
    return *maybeTripCnt;
  }
  return currGs;
}

void setValidStaticGPUConfigAttr(func::FuncOp func, ArrayRef<int64_t> bs,
                                 ArrayRef<int64_t> gs, int64_t coarsenFactor) {

  // handle block and grid sizes
  SmallVector<int64_t> toGPUSizes;

  // read attrs
  if (auto arrayAttr = func->getAttrOfType<ArrayAttr>(getToGPUAttrName())) {
    for (auto attr : arrayAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        toGPUSizes.push_back(intAttr.getInt());
      } else {
        toGPUSizes.push_back(1);
      }
    }
  } else {
    toGPUSizes.insert(toGPUSizes.end(), bs.begin(), bs.end());
    toGPUSizes.insert(toGPUSizes.end(), gs.begin(), gs.end());
  }

  SmallVector<Attribute> toGPUAttrs;
  auto ctx = func.getContext();

  for (size_t i = 0; i < 3; ++i) {
    if (i >= toGPUSizes.size()) {
      toGPUSizes.push_back(1);
    }

    if (toGPUSizes[i] <= 0) {
      toGPUSizes[i] = 1;
    }
  }

  // estimate maxGridSizes if possible
  SmallVector<int64_t> maxGridSizes = {0, 0, 0};
  // collect loops from inner to outer
  bool firstCheck = true;
  auto isSuitableConfig = [&]() -> bool {
    if (llvm::all_of(maxGridSizes, [](int64_t val) { return val == 0; })) {
      return false;
    }
    int64_t totalGridSize = 1;
    for (auto v : maxGridSizes) {
      if (v != 0)
        totalGridSize *= v;
    }
    int64_t totalBlockSize = 1;
    for (size_t i = 0; i < 3; ++i) {
      totalBlockSize *= toGPUSizes[i];
    }
    if (totalGridSize < kGridTileNumThreshold &&
        totalBlockSize >= kWarpSize * 2) {
      return false;
    }
    return true;
  };
  while (!isSuitableConfig()) {
    if (!firstCheck) {
      for (int64_t i = 2; i >= 0; --i) {
        if (toGPUSizes[i] >= 2) {
          toGPUSizes[i] /= 2;
          break;
        }
      }
    }
    firstCheck = false;
    maxGridSizes = {0, 0, 0};

    func.walk([&](LoopLikeOpInterface loopLike) {
      if (loopLike->hasAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
        auto coarsen =
            loopLike->hasAttrOfType<UnitAttr>(getCoarsenSIMTAttrName());
        int64_t factor = coarsen ? coarsenFactor : 1;

        auto strAttr =
            loopLike->getAttrOfType<StringAttr>(getLoopToSIMTAttrName());

        if (strAttr.getValue() == getLinearIdXName()) {
          maxGridSizes[0] = estimateGridSize(loopLike, maxGridSizes[0],
                                             toGPUSizes[0] * factor);
        } else if (strAttr.getValue() == getLinearIdYName()) {
          maxGridSizes[1] = estimateGridSize(loopLike, maxGridSizes[1],
                                             toGPUSizes[1] * factor);
        } else if (strAttr.getValue() == getLinearIdZName()) {
          maxGridSizes[2] = estimateGridSize(loopLike, maxGridSizes[2],
                                             toGPUSizes[2] * factor);
        } else if (strAttr.getValue() == getBlockIdXName()) {
          maxGridSizes[0] = estimateGridSize(loopLike, maxGridSizes[0], factor);
        } else if (strAttr.getValue() == getBlockIdYName()) {
          maxGridSizes[1] = estimateGridSize(loopLike, maxGridSizes[1], factor);
        } else if (strAttr.getValue() == getBlockIdZName()) {
          maxGridSizes[2] = estimateGridSize(loopLike, maxGridSizes[2], factor);
        }
      }
    });
  }

  int64_t threshold = kGridTileNumThreshold * kNumWave;
  for (size_t i = 0; i < maxGridSizes.size(); ++i) {
    if (maxGridSizes[i] > threshold) {
      maxGridSizes[i] = threshold;
    } else {
      threshold /= maxGridSizes[i];
      break;
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    auto attr = IntegerAttr::get(IntegerType::get(ctx, 32), toGPUSizes[i]);
    toGPUAttrs.push_back(attr);
  }
  for (size_t i = 0; i < 3; ++i) {
    size_t j = i + 3;
    // if no estimation use suggested attr value
    if (maxGridSizes[i] == 0) {
      if (j < toGPUSizes.size() && toGPUSizes[j] > 0) {
        maxGridSizes[i] = toGPUSizes[j];
      } else {
        maxGridSizes[i] = 1;
      }
    }

    auto attr = IntegerAttr::get(IntegerType::get(ctx, 32), maxGridSizes[i]);
    toGPUAttrs.push_back(attr);
  }

  func->setAttr(getToGPUAttrName(), ArrayAttr::get(ctx, toGPUAttrs));
}

static std::optional<Attribute>
getMaxGPUConfigAttr(Operation *op, KernelDim3 bs, KernelDim3 gs) {
  std::optional<Attribute> attr = std::nullopt;
  if (auto threadId = dyn_cast<gpu::ThreadIdOp>(op)) {
    if (threadId.getDimension() == gpu::Dimension::x) {
      attr = getAttrFromConstantLike(bs.x);
    } else if (threadId.getDimension() == gpu::Dimension::y) {
      attr = getAttrFromConstantLike(bs.y);
    } else {
      attr = getAttrFromConstantLike(bs.z);
    }
  } else if (auto blockId = dyn_cast<gpu::BlockIdOp>(op)) {
    if (blockId.getDimension() == gpu::Dimension::x) {
      attr = getAttrFromConstantLike(gs.x);
    } else if (blockId.getDimension() == gpu::Dimension::y) {
      attr = getAttrFromConstantLike(gs.y);
    } else {
      attr = getAttrFromConstantLike(gs.z);
    }
  }
  if (!attr)
    return std::nullopt;
  auto intAttr = attr->cast<IntegerAttr>();
  return IntegerAttr::get(intAttr.getType(), intAttr.getInt() - 1);
}

static IRMapping getMaxGPUConfigAttrs(OpBuilder &b, gpu::GPUFuncOp gFunc,
                                      KernelDim3 bs, KernelDim3 gs) {
  IRMapping bvm;
  for (auto threadId : getOpsNested<gpu::ThreadIdOp>(gFunc)) {
    auto attr = getMaxGPUConfigAttr(threadId, bs, gs);
    if (!attr)
      continue;

    b.setInsertionPoint(threadId);
    Value newConst = arith::ConstantOp::materialize(
        b, *attr, threadId.getType(), threadId.getLoc());
    bvm.map(threadId.getResult(), newConst);
  }

  for (auto blockId : getOpsNested<gpu::BlockIdOp>(gFunc)) {
    auto attr = getMaxGPUConfigAttr(blockId, bs, gs);
    if (!attr)
      continue;
    b.setInsertionPoint(blockId);
    Value newConst = arith::ConstantOp::materialize(b, *attr, blockId.getType(),
                                                    blockId.getLoc());
    bvm.map(blockId.getResult(), newConst);
  }
  return bvm;
}

static void simplifyGuards(gpu::GPUFuncOp gFunc, gpu::LaunchFuncOp launch) {
  OpBuilder builder(gFunc.getContext());
  KernelDim3 bs = launch.getBlockSizeOperandValues();

  // create bvm from static launch values
  IRMapping bvm =
      getMaxGPUConfigAttrs(builder, gFunc, launch.getBlockSizeOperandValues(),
                           launch.getGridSizeOperandValues());

  // check whether condition can be simplified
  for (auto ifOp : getOpsNested<scf::IfOp>(gFunc)) {
    auto cond = ifOp.getCondition();
    if (auto cmp = cond.getDefiningOp<arith::CmpIOp>()) {
      SmallVector<OpFoldResult> foldResults;
      auto isFold = deepFold(cmp, bvm, foldResults);
      if (succeeded(isFold) && foldResults.size() == 1) {
        auto attr = foldResults[0].dyn_cast<Attribute>();
        // check it is true
        // if so, override the condition into true
        if (attr.cast<IntegerAttr>().getInt() != 0) {
          builder.setInsertionPoint(ifOp);
          Value newConst = arith::ConstantOp::materialize(
              builder, attr, cmp.getType(), cmp.getLoc());
          ifOp.setOperand(newConst);
        }
      }
    }
  }
}

struct ConvertFuncToGPUPass
    : public ConvertFuncToGPUBase<ConvertFuncToGPUPass> {
  ConvertFuncToGPUPass(ArrayRef<int64_t> bs, ArrayRef<int64_t> gs,
                       const std::string &name)
      : ConvertFuncToGPUBase() {
    this->blockSizes = bs;
    this->gridSizes = gs;
    this->moduleName = name;
  }

  void runOnOperation() final {

    // early termination if no anchor or no moduleName
    if (moduleName.empty()) {
      return;
    }

    ModuleOp m = getOperation();
    SmallVector<func::FuncOp> funcCollector;

    // collect all anchored function
    for (auto func : m.getOps<func::FuncOp>()) {
      if (func->hasAttr(getToGPUAttrName())) {
        // TODO: configurable coarsen factor
        setValidStaticGPUConfigAttr(func, blockSizes, gridSizes,
                                    COARSEN_FACTOR);
        funcCollector.push_back(func);
      }
    }

    // early termination if no anchored func
    if (funcCollector.empty()) {
      return;
    }

    auto gm = getOrCreateGPUModule(m, moduleName);
    SymbolTable gmTable(gm);

    OpBuilder builder(gm.getContext());
    SmallVector<std::pair<gpu::GPUFuncOp, gpu::LaunchFuncOp>> gFuncAndLaunchs;
    // create GPUFuncOp and gpu::LaunchFunc
    for (auto func : funcCollector) {
      // perform hoist first
      hoistUpOpsInFuncLike(func, isHoistUpOp);

      rewriteFuncImpl(builder, func);

      // create GPUFunc
      SmallVector<Value> args;
      auto gpuFunc = cloneFuncToGPUFunc(builder, func, gm, args);

      // gFuncs.push_back(gpuFunc);
      gmTable.insert(gpuFunc);

      // create GPULaunchFunc
      auto launch = rewriteToGPULaunchFuncImpl(builder, func, args, gpuFunc);
      gFuncAndLaunchs.emplace_back(gpuFunc, launch);

      // remove attr
      func->removeAttr(getToGPUAttrName());
    }

    // set gpu.container_module
    m->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
               UnitAttr::get(m.getContext()));

    // perform clean ups
    {
      OpPassManager pm(m.getOperationName());
      addCleanUpPassPipeline(pm);
      addMultiCSEPipeline(pm, 2);

      if (mlir::failed(runPipeline(pm, m))) {
        signalPassFailure();
      }
    }

    // perform simplifyGuards
    {
      for (auto &p : gFuncAndLaunchs) {
        simplifyGuards(p.first, p.second);
      }
    }

    // perform clean ups again
    {
      OpPassManager pm(m.getOperationName());
      addCleanUpPassPipeline(pm);
      addMultiCSEPipeline(pm, 2);
      if (mlir::failed(runPipeline(pm, m))) {
        signalPassFailure();
      }
    }

    // perform CMAE
    {
      for (auto &p : gFuncAndLaunchs) {
        runCMAEInFuncLike(p.first);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToGPUPass(ArrayRef<int64_t> bs, ArrayRef<int64_t> gs,
                                 const std::string &name) {

  return std::make_unique<ConvertFuncToGPUPass>(bs, gs, name);
}
