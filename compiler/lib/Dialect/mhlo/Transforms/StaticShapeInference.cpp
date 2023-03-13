//===- StaticShapeInference.cpp -------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/StaticShapeInference.h"

#include "byteir/Analysis/ShapeAnalysis.h"
#include "byteir/Dialect/mhlo/Analysis/ShapeAnalysis.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::shape_analysis;
using namespace mlir::dataflow;

#define DEBUG_TYPE "static-shape-infer"

namespace {

LogicalResult
constructNewArgumentTypes(func::FuncOp funcOp,
                          SmallVectorImpl<Type> &newArgumentTypes) {
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    Type origType = funcOp.getArgumentTypes()[i];

    auto origRankedType = origType.dyn_cast<RankedTensorType>();
    if (!origRankedType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Argument " << i << "is not of type RankedTensorType.\n");
      return failure();
    }

    if (origRankedType.hasStaticShape()) {
      newArgumentTypes.push_back(origType);
      continue;
    }

    auto staticShapeAttr =
        funcOp.getArgAttrOfType<ArrayAttr>(i, getStaticShapeAttrName());
    if (!staticShapeAttr) {
      LLVM_DEBUG(llvm::dbgs() << "Argument " << i
                              << "doesn't have either static shape or "
                                 "static shape attribute.\n");
      return failure();
    }

    SmallVector<int64_t> staticShape = llvm::to_vector(
        llvm::map_range(staticShapeAttr.getAsRange<IntegerAttr>(),
                        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));

    if (int64_t(staticShape.size()) != origRankedType.getRank()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Argument " << i << "'s rank: " << origRankedType.getRank()
                 << ", is not equal to static shape's rank: "
                 << staticShape.size() << "\n");
      return failure();
    }

    Type newType = origRankedType.clone(staticShape);
    newArgumentTypes.push_back(newType);
  }
  return success();
}

} // namespace

LogicalResult mlir::runStaticShapeInfer(func::FuncOp funcOp,
                                        bool overrideShape) {
  MLIRContext *context = funcOp.getContext();
  SmallVector<Type> newArgumentTypes;
  if (failed(constructNewArgumentTypes(funcOp, newArgumentTypes))) {
    return failure();
  }

  func::ReturnOp returnOp;
  {
    ValueTypeModificatoinRAII valueTypeModification;
    for (auto &&pi : llvm::zip(funcOp.getArguments(), newArgumentTypes)) {
      valueTypeModification.Push(std::get<0>(pi), std::get<1>(pi));
    }

    DataFlowSolver solver;
    solver.load<MhloShapeAnalysis>();
    solver.load<MhloShapeValueAnalysis>();
    solver.load<DeadCodeAnalysis>();
    if (failed(solver.initializeAndRun(funcOp)))
      return failure();

    // Early return when not override shape
    if (!overrideShape) {
      return success();
    }

    funcOp->walk([&](Operation *op) {
      for (auto &&it : op->getResults()) {
        auto originalType = it.getType().dyn_cast<ShapedType>();
        if (!originalType || originalType.hasStaticShape())
          continue;

        ShapedType newType;
        if (auto lattice = solver.lookupState<ShapeLattice>(it)) {
          if (!lattice->getValue().isUninitialized())
            newType = lattice->getValue().getType();
        }

        if (!newType || !newType.hasRank())
          continue;

        it.setType(newType);
      }
      if (llvm::isa<func::ReturnOp>(op)) {
        returnOp = cast<func::ReturnOp>(op);
      }
    });
  }
  assert(returnOp && "there must be return op in func");

  auto newFuncRetTypes = llvm::to_vector(returnOp.getOperandTypes());
  auto newFuncType =
      FunctionType::get(context, newArgumentTypes, newFuncRetTypes);
  funcOp.setType(newFuncType);
  for (auto elem : llvm::zip(funcOp.front().getArguments(), newArgumentTypes)) {
    std::get<0>(elem).setType(std::get<1>(elem));
  }
  return success();
}

namespace {

struct StaticShapeInferencePass
    : public StaticShapeInferenceBase<StaticShapeInferencePass> {

  StaticShapeInferencePass(bool overrideShape)
      : StaticShapeInferenceBase<
            StaticShapeInferencePass>::StaticShapeInferenceBase() {
    this->overrideShape = overrideShape;
    registerAllMhloInferReturnTypeComponents();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (failed(runStaticShapeInfer(funcOp, this->overrideShape))) {
      return signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createStaticShapeInferencePass(bool overrideShape) {
  return std::make_unique<StaticShapeInferencePass>(overrideShape);
}
