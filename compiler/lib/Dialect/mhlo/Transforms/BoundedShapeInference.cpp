//===- BoundedShapeInference.cpp ------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/BoundedShapeInference.h"

#include "byteir/Dialect/mhlo/Analysis/ShapeAnalysis.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "byteir/Utils/TypeUtils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <string>
#include <vector>

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::shape_analysis;
using namespace mlir::dataflow;

#define DEBUG_TYPE "bounded-shape-infer"

namespace {

LogicalResult constructNewArgumentTypes(func::FuncOp funcOp,
                                        SmallVectorImpl<Type> &newArgumentTypes,
                                        SmallVectorImpl<Type> &newFuncArgTypes,
                                        OpBuilder &builder) {
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
      newFuncArgTypes.push_back(origType);
      continue;
    }

    auto boundedShapeAttr =
        funcOp.getArgAttrOfType<ArrayAttr>(i, getBoundedShapeAttrName());
    if (!boundedShapeAttr) {
      LLVM_DEBUG(llvm::dbgs() << "Argument " << i
                              << "doesn't have either static shape or "
                                 "bounded shape attribute.\n");
      return failure();
    }

    SmallVector<int64_t> boundedShape = llvm::to_vector(
        llvm::map_range(boundedShapeAttr.getAsRange<IntegerAttr>(),
                        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));

    if (int64_t(boundedShape.size()) != origRankedType.getRank()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Argument " << i << "'s rank: " << origRankedType.getRank()
                 << ", is not equal to bounded shape's rank: "
                 << boundedShape.size() << "\n");
      return failure();
    }

    auto typeWithEncoding = appendTensorEncodingAttr(
        origRankedType,
        builder.getNamedAttr(getBoundedShapeAttrName(),
                             builder.getI64ArrayAttr(boundedShape)));
    newFuncArgTypes.push_back(typeWithEncoding);
    Type newType = origRankedType.clone(boundedShape);
    newArgumentTypes.push_back(newType);
  }
  return success();
}

struct BoundedShapeInferencePass
    : public BoundedShapeInferenceBase<BoundedShapeInferencePass> {

  BoundedShapeInferencePass()
      : BoundedShapeInferenceBase<
            BoundedShapeInferencePass>::BoundedShapeInferenceBase() {
    registerAllMhloInferReturnTypeComponents();
    registerAllMhloInferBoundedReturnTypeComponents();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp);

    SmallVector<Type> newArgumentTypes;
    SmallVector<Type> newFuncArgTypes;
    if (failed(constructNewArgumentTypes(funcOp, newArgumentTypes,
                                         newFuncArgTypes, builder))) {
      return signalPassFailure();
    }

    func::ReturnOp returnOp;
    {
      ValueTypeModificatoinRAII valueTypeModification;
      for (auto &&pi : llvm::zip(funcOp.getArguments(), newArgumentTypes)) {
        valueTypeModification.Push(std::get<0>(pi), std::get<1>(pi));
      }

      DataFlowSolver solver;
      solver.load<MhloBoundedShapeAnalysis>();
      solver.load<MhloShapeValueAnalysis>();
      solver.load<MhloBoundedValueAnalysis>();
      solver.load<DeadCodeAnalysis>();
      if (failed(solver.initializeAndRun(funcOp)))
        return signalPassFailure();

      funcOp->walk([&](Operation *op) {
        for (auto &&it : op->getResults()) {
          auto originalType = it.getType().dyn_cast<ShapedType>();
          if (!originalType || originalType.hasStaticShape())
            continue;

          ShapedType newType;
          if (auto *lattice = solver.lookupState<ShapeLattice>(it)) {
            if (!lattice->getValue().isUninitialized()) {
              newType = lattice->getValue().getType().dyn_cast<ShapedType>();
            }
          }

          if (!newType || !newType.hasRank())
            continue;

          auto tx = it.getType().dyn_cast<RankedTensorType>();

          ArrayRef<int64_t> shape = newType.getShape();
          OpBuilder builder(op);
          if (tx) {
            auto typeWithEncoding = appendTensorEncodingAttr(
                tx, builder.getNamedAttr(getBoundedShapeAttrName(),
                                         builder.getI64ArrayAttr(shape)));
            it.setType(typeWithEncoding);
          }
        }
        if (isa<func::ReturnOp>(op)) {
          returnOp = cast<func::ReturnOp>(op);
        }
      });
    }
    assert(returnOp && "there must be return op in func");

    auto newFuncRetTypes = llvm::to_vector(returnOp.getOperandTypes());
    auto newFuncType =
        FunctionType::get(&getContext(), newFuncArgTypes, newFuncRetTypes);
    funcOp.setType(newFuncType);
    for (auto elem :
         llvm::zip(funcOp.front().getArguments(), newFuncArgTypes)) {
      std::get<0>(elem).setType(std::get<1>(elem));
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createBoundedShapeInferencePass() {
  return std::make_unique<BoundedShapeInferencePass>();
}
