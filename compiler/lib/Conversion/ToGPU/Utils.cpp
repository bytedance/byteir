//===- Utils.cpp -------------------------------------------------- C++ -*-===//
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

#include "byteir/Conversion/ToGPU/Utils.h"

#include "byteir/Analysis/Alias.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

using namespace byteir;
using namespace llvm;
using namespace mlir;
using namespace mlir::gpu;

GPUModuleOp mlir::getOrCreateGPUModule(ModuleOp m, StringRef moduleName) {
  for (auto &op : m.getBody()->without_terminator()) {
    if (auto gm = dyn_cast<gpu::GPUModuleOp>(op)) {
      if (gm.getName() == moduleName) {
        return gm;
      }
    }
  }

  // if not found, create one
  OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
  auto gm = builder.create<GPUModuleOp>(m.getLoc(), moduleName);
  return gm;
}

namespace {
static bool isAliasOp(Operation &op) {
  return isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::ReshapeOp>(
      op);
};

} // namespace

bool mlir::isGPUGlobalAlloc(Operation &op) {
  if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
    auto memrefTy = alloc.getResult().getType().dyn_cast<MemRefType>();
    auto spaceAttr = memrefTy.getMemorySpace();
    if (spaceAttr == nullptr) {
      return true;
    }

    if (auto intAttr = spaceAttr.dyn_cast<IntegerAttr>()) {
      return intAttr.getInt() == 1;
    }
  }

  return false;
}

bool mlir::isGPUGlobalAlloc(Operation *op) {
  if (op == nullptr)
    return false;
  return isGPUGlobalAlloc(*op);
}

gpu::GPUFuncOp mlir::cloneFuncToGPUFunc(OpBuilder &builder, func::FuncOp func,
                                        gpu::GPUModuleOp gm,
                                        SmallVectorImpl<Value> &args) {

  // handle aliasOp
  SmallVector<Value> initialVals;
  // put func's arguments in initialVals of AliasAnalysis
  for (auto val : func.getArguments()) {
    initialVals.push_back(val);
  }

  // put global alloc in initialVals of AliasAnalysis
  for (auto alloc : func.getOps<memref::AllocOp>()) {
    if (isGPUGlobalAlloc(alloc)) {
      initialVals.push_back(alloc.getResult());
    }
  }

  Region &funcBody = func.getBody();
  Block &funcEntryBlock = funcBody.front();
  // perform AliasAnalysis
  AliasAnalysis aliasAnalysis(&funcEntryBlock, initialVals, isAliasOp);
  aliasAnalysis.runOnBlock();

  args.insert(args.end(), func.getArguments().begin(),
              func.getArguments().end());
  // create a new input Types
  // clone old inputTypes first
  SmallVector<Type> inputTypes(func.getFunctionType().getInputs().begin(),
                               func.getFunctionType().getInputs().end());

  // append return types if not alias
  auto ret = funcEntryBlock.getTerminator();

  SmallPtrSet<Operation *, 4> usedGlobalAlloc;
  SmallDenseMap<int, int> uniqueIndexMap;
  int offset = func.getNumArguments();
  for (auto retVal : ret->getOperands()) {
    auto leader = aliasAnalysis.getLeaderIndex(retVal);

    auto leaderVal = aliasAnalysis.values[leader];
    // check if leaderVal is from a new alloc
    if (static_cast<unsigned>(leader) >= func.getNumArguments() &&
        uniqueIndexMap.count(leader) == 0) {
      inputTypes.push_back(leaderVal.getType());
      args.push_back(leaderVal);
      usedGlobalAlloc.insert(leaderVal.getDefiningOp());
      uniqueIndexMap.try_emplace(leader, offset++);
    }
  }

  auto ctx = func.getContext();
  auto newFuncType = FunctionType::get(ctx, inputTypes, {});

  builder.setInsertionPointToStart(gm.getBody());

  // handle workgroupAttributions aka shared memory
  auto workgroupAttr = wrapIntegerMemorySpace(/*space=*/3, ctx);

  SmallVector<Value> oldWorkgroupValue;
  SmallVector<Type> workgroupAttrTypes;
  for (auto alloc : func.getOps<memref::AllocOp>()) {
    if (!usedGlobalAlloc.contains(alloc)) {
      oldWorkgroupValue.push_back(alloc.getResult());
      auto neweTy = cloneMemRefTypeWithMemSpace(
          alloc.getResult().getType().dyn_cast<MemRefType>(), workgroupAttr);
      workgroupAttrTypes.push_back(neweTy);
    }
  }

  auto gpuFunc = builder.create<gpu::GPUFuncOp>(
      gm.getLoc(), func.getName(), newFuncType, workgroupAttrTypes);

  IRMapping bvm;
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    bvm.map(func.getArgument(i), gpuFunc.getArgument(i));
  }

  for (unsigned i = 0; i < oldWorkgroupValue.size(); ++i) {
    bvm.map(oldWorkgroupValue[i], gpuFunc.getWorkgroupAttributions()[i]);
  }

  for (auto it : uniqueIndexMap) {
    auto leaderVal = aliasAnalysis.values[it.first];
    bvm.map(leaderVal, gpuFunc.getArgument(it.second));
  }

  gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                   builder.getUnitAttr());

  Region &gpuFuncBody = gpuFunc.getBody();

  Block &gpuEntryBlock = gpuFuncBody.front();

  builder.setInsertionPointToStart(&gpuEntryBlock);
  for (auto &op : funcEntryBlock.without_terminator()) {
    if (!isGPUGlobalAlloc(op)) {
      builder.clone(op, bvm);
    }
  }

  // create a terminator
  builder.create<gpu::ReturnOp>(gpuFunc.getLoc());

  return gpuFunc;
}
