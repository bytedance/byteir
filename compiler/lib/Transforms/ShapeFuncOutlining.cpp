//===- ShapeFuncOutlining.cpp ------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/ShapeFuncOutlining.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"
#include <queue>

#include "./PassDetail.h"

#define LAUNCH_CONFIG_NUM 7

using namespace mlir;

namespace {

static SmallVector<Operation *, 8>
getBackwardSliceOfOutputs(const SmallVector<Value> &outputs) {
  llvm::DenseSet<Operation *> opSet;
  llvm::SetVector<Operation *> backwardSlice;
  for (Value result : outputs) {
    llvm::SetVector<Operation *> backwardSlice;
    BackwardSliceOptions options;
    options.inclusive = true;
    getBackwardSlice(result, &backwardSlice, options);
    opSet.insert(backwardSlice.begin(), backwardSlice.end());
  }
  SmallVector<Operation *, 8> backwardOps(opSet.begin(), opSet.end());
  return backwardOps;
}

static bool isOpOnlyUseOperandMeta(Operation *op) {
  return llvm::isa<memref::DimOp, memref::RankOp>(op);
}

static bool
isClusterOnlyUseInputMeta(const SmallVector<Operation *, 8> &cluster) {
  auto inputs = getInputsOfCluster(cluster);
  llvm::DenseSet<Value> inputSet(inputs.begin(), inputs.end());
  for (auto op : cluster) {
    bool useInput = false;
    for (auto operand : op->getOperands()) {
      if (inputSet.find(operand) != inputSet.end()) {
        useInput = true;
        break;
      }
    }
    if (useInput && !isOpOnlyUseOperandMeta(op))
      return false;
  }
  return true;
}

static std::pair<func::FuncOp, byre::ComputeShapeOp>
createShapeFuncForSpecificOutputs(OpBuilder &builder,
                                  const SmallVector<Value> &outputs) {
  static size_t shapeFuncCnt = 0;
  llvm::SmallVector<Operation *, 8> shapeComputationOps =
      getBackwardSliceOfOutputs(outputs);
  mlir::computeTopologicalSorting(shapeComputationOps);
  auto inputs = getInputsOfCluster(shapeComputationOps);

  SmallVector<Location, 8> locations;
  for (Operation *op : shapeComputationOps) {
    locations.push_back(op->getLoc());
  }

  Location fusedLoc =
      FusedLoc::get(shapeComputationOps.back()->getContext(), locations);

  SmallVector<Type, 4> inputTypes, outputTypes;
  for (Value v : inputs) {
    inputTypes.emplace_back(v.getType());
  }

  for (Value v : outputs) {
    outputTypes.emplace_back(v.getType());
  }

  auto funcType = builder.getFunctionType(inputTypes, outputTypes);
  builder.setInsertionPoint(shapeComputationOps[0]->getParentOp());
  std::string funcName = "ShapeComputaionFunc" + std::to_string(shapeFuncCnt);
  shapeFuncCnt += 1;

  func::FuncOp funcOp =
      builder.create<func::FuncOp>(fusedLoc, funcName, funcType);
  funcOp.setPrivate();

  // clone shape computation ops to shape func
  Block *block = funcOp.addEntryBlock();
  builder.setInsertionPoint(block, block->end());
  IRMapping bvm;
  for (auto inputAndArg : llvm::zip(inputs, funcOp.getArguments())) {
    bvm.map(std::get<0>(inputAndArg), std::get<1>(inputAndArg));
  }
  for (Operation *op : shapeComputationOps) {
    builder.clone(*op, bvm);
  }

  llvm::SmallVector<Value, 6> funcReturns;
  for (Value out : outputs) {
    funcReturns.emplace_back(bvm.lookupOrDefault(out));
  }
  builder.create<func::ReturnOp>(fusedLoc, funcReturns);

  // create ComputeShapeOp
  builder.setInsertionPoint(shapeComputationOps[0]);
  auto shape_fn = funcOp.getName();
  auto computeShapeOp = builder.create<byre::ComputeShapeOp>(
      fusedLoc, outputTypes, shape_fn, inputs);

  // replace user
  for (auto outputAndResult : llvm::zip(outputs, computeShapeOp.getResults())) {
    Value output, CallResult;
    std::tie(output, CallResult) = outputAndResult;
    for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
      use.set(CallResult);
    }
  }

  // reversal order, avoid erase a op still in use.
  std::reverse(shapeComputationOps.begin(), shapeComputationOps.end());
  for (Operation *op : shapeComputationOps) {
    op->erase();
  }

  // attach device attr to cpu
  computeShapeOp->setAttr("device",
                          StringAttr::get(funcOp.getContext(), "cpu"));

  funcOp->setAttr(getByteIRShapeFuncAttrName(),
                  UnitAttr::get(funcOp.getContext()));

  return {funcOp, computeShapeOp};
}

struct ShapeFuncOutliningPass
    : public ShapeFuncOutliningBase<ShapeFuncOutliningPass> {

  ShapeFuncOutliningPass(llvm::StringRef entryFuncName)
      : ShapeFuncOutliningBase() {
    this->entryFuncName = entryFuncName.str();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    func::FuncOp funcOp = m.lookupSymbol<func::FuncOp>(this->entryFuncName);
    if (!funcOp) {
      return;
    }
    OpBuilder builder(funcOp);

    llvm::DenseSet<Value> outputsSet;
    for (auto &block : funcOp.getBlocks()) {
      for (auto &op : block.without_terminator()) {
        if (auto allocOp = llvm::dyn_cast<memref::AllocOp>(op)) {
          if (allocOp.getDynamicSizes().size() > 0) {
            outputsSet.insert(allocOp.getDynamicSizes().begin(),
                              allocOp.getDynamicSizes().end());
          }
        } else if (auto byreOp = llvm::dyn_cast<byre::ComputeOp>(op)) {
          if (byreOp->hasAttr(byre::getByreDynamicLaunchConfigAttrName())) {
            size_t inputNum = byreOp.getOperands().size();
            for (size_t i = inputNum - LAUNCH_CONFIG_NUM; i < inputNum; ++i) {
              outputsSet.insert(byreOp.getOperand(i));
            }
          }
        }
      }
    }

    if (outputsSet.size() > 0) {
      llvm::SmallVector<Value> outputs(outputsSet.begin(), outputsSet.end());
      llvm::SmallVector<Operation *, 8> shapeComputationOps =
          getBackwardSliceOfOutputs(outputs);
      if (isClusterOnlyUseInputMeta(shapeComputationOps)) {
        (void)createShapeFuncForSpecificOutputs(builder, outputs);
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createShapeFuncOutliningPass(llvm::StringRef entryFuncName) {
  return std::make_unique<ShapeFuncOutliningPass>(entryFuncName.str());
}
