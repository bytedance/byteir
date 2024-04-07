//===- ShapeInferUtil.cpp -------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Transforms/ShapeReification.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <string>

using namespace mlir;

#define DEBUG_TYPE "shape-infer-util"

//===----------------------------------------------------------------------===//
// ReifyReturnTypeShapes Registration
//===----------------------------------------------------------------------===//

static llvm::StringMap<ReifyReturnTypeShapes> &
getReifyReturnTypeShapesRegistry() {
  static llvm::StringMap<ReifyReturnTypeShapes> reifyReturnTypeShapesRegistry;
  return reifyReturnTypeShapesRegistry;
}

/// Register the given ReifyReturnTypeShapes function.
static void
registerReifyReturnTypeShapes(StringRef name,
                              const ReifyReturnTypeShapes &function) {
  auto &reifyReturnTypeShapesRegistry = getReifyReturnTypeShapesRegistry();
  if (reifyReturnTypeShapesRegistry.find(name) !=
      reifyReturnTypeShapesRegistry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing ReifyReturnTypeShapes function");
  assert(function &&
         "Attempting to register an empty ReifyReturnTypeShapes function");
  reifyReturnTypeShapesRegistry[name] = function;
}

ReifyReturnTypeShapesRegistration::ReifyReturnTypeShapesRegistration(
    StringRef name, const ReifyReturnTypeShapes &function) {
  registerReifyReturnTypeShapes(name, function);
}

ReifyReturnTypeShapes mlir::reifyReturnTypeShapes(llvm::StringRef name) {
  auto &reifyReturnTypeShapesRegistry = getReifyReturnTypeShapesRegistry();
  auto it = reifyReturnTypeShapesRegistry.find(name);
  if (it != reifyReturnTypeShapesRegistry.end())
    return it->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// InsertShapeConstraint Registration
//===----------------------------------------------------------------------===//

static llvm::StringMap<InsertShapeConstraint> &
getInsertShapeConstraintRegistry() {
  static llvm::StringMap<InsertShapeConstraint> insertShapeConstraintRegistry;
  return insertShapeConstraintRegistry;
}

/// Register the given ReifyReturnTypeShapes function.
static void
registerInsertShapeConstraint(StringRef name,
                              const InsertShapeConstraint &function) {
  auto &insertShapeConstraintRegistry = getInsertShapeConstraintRegistry();
  if (insertShapeConstraintRegistry.find(name) !=
      insertShapeConstraintRegistry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing InsertShapeConstraint function");
  assert(function &&
         "Attempting to register an empty InsertShapeConstraint function");
  insertShapeConstraintRegistry[name] = function;
}

InsertShapeConstraintRegistration::InsertShapeConstraintRegistration(
    StringRef name, const InsertShapeConstraint &function) {
  registerInsertShapeConstraint(name, function);
}

InsertShapeConstraint mlir::insertShapeConstraint(llvm::StringRef name) {
  auto &insertShapeConstraintRegistry = getInsertShapeConstraintRegistry();
  auto it = insertShapeConstraintRegistry.find(name);
  if (it != insertShapeConstraintRegistry.end())
    return it->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// InferBoundedReturnTypeComponents Registration
//===----------------------------------------------------------------------===//

static llvm::StringMap<InferBoundedReturnTypeComponents> &
getInferBoundedReturnTypeComponentsRegistry() {
  static llvm::StringMap<InferBoundedReturnTypeComponents>
      InferBoundedReturnTypeComponentsRegistry;
  return InferBoundedReturnTypeComponentsRegistry;
}

static void registerInferBoundedReturnTypeComponents(
    StringRef name, const InferBoundedReturnTypeComponents &function) {
  auto &registry = getInferBoundedReturnTypeComponentsRegistry();
  if (registry.find(name) != registry.end())
    llvm::report_fatal_error("Attempting to overwrite an existing "
                             "InferBoundedReturnTypeComponents function");
  assert(function && "Attempting to register an empty "
                     "InferBoundedReturnTypeComponents function");
  registry[name] = function;
}

InferBoundedReturnTypeComponentsRegistration::
    InferBoundedReturnTypeComponentsRegistration(
        StringRef name, const InferBoundedReturnTypeComponents &function) {
  registerInferBoundedReturnTypeComponents(name, function);
}

InferBoundedReturnTypeComponents
mlir::inferBoundedReturnTypeComponents(llvm::StringRef name) {
  auto &registry = getInferBoundedReturnTypeComponentsRegistry();
  auto it = registry.find(name);
  if (it != registry.end())
    return it->second;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// InferReturnTypeComponents Registration
//===----------------------------------------------------------------------===//

static llvm::StringMap<InferReturnTypeComponents> &
getInferReturnTypeComponentsRegistry() {
  static llvm::StringMap<InferReturnTypeComponents>
      InferReturnTypeComponentsRegistry;
  return InferReturnTypeComponentsRegistry;
}

static void
registerInferReturnTypeComponents(StringRef name,
                                  const InferReturnTypeComponents &function) {
  auto &registry = getInferReturnTypeComponentsRegistry();
  if (registry.find(name) != registry.end())
    llvm::report_fatal_error("Attempting to overwrite an existing "
                             "InferReturnTypeComponents function");
  assert(function && "Attempting to register an empty "
                     "InferReturnTypeComponents function");
  registry[name] = function;
}

InferReturnTypeComponentsRegistration::InferReturnTypeComponentsRegistration(
    StringRef name, const InferReturnTypeComponents &function) {
  registerInferReturnTypeComponents(name, function);
}

InferReturnTypeComponents
mlir::inferReturnTypeComponents(llvm::StringRef name) {
  auto &registry = getInferReturnTypeComponentsRegistry();
  auto it = registry.find(name);
  if (it != registry.end())
    return it->second;
  return nullptr;
}

namespace {

SmallVector<Operation *> collectAllOpsForReturn(Operation *retOp) {
  llvm::DenseSet<Operation *> visitedOp;
  std::queue<Operation *> opQueue;

  opQueue.push(retOp);
  while (!opQueue.empty()) {
    auto frontOp = opQueue.front();
    opQueue.pop();
    if (visitedOp.find(frontOp) != visitedOp.end()) {
      continue;
    }
    visitedOp.insert(frontOp);
    for (Value operand : frontOp->getOperands()) {
      if (!operand.getDefiningOp()) {
        continue;
      }
      if (Operation *defOp = operand.getDefiningOp()) {
        opQueue.push(defOp);
      }
    }
  }
  visitedOp.erase(retOp);
  return SmallVector<Operation *>(visitedOp.begin(), visitedOp.end());
}

bool deduceFromFuncArgShape(Value value) {
  if (value.isa<BlockArgument>()) {
    return false;
  }

  auto defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  if (isa<arith::ConstantIndexOp, arith::ConstantOp>(defOp)) {
    return true;
  }

  if (isa<tensor::DimOp, shape::ShapeOfOp>(defOp)) {
    auto operand = defOp->getOperand(0);
    if (operand.isa<BlockArgument>()) {
      return true;
    }
    return false;
  }

  for (Value &&operand : defOp->getOperands()) {
    if (!deduceFromFuncArgShape(operand)) {
      return false;
    }
  }
  return true;
}

LogicalResult reifyCallOp(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications) {
  OpBuilder::InsertionGuard guard(builder);
  auto callOp = dyn_cast<func::CallOp>(op);
  if (!callOp) {
    return failure();
  }

  ModuleOp moduleOp = op->getParentRegion()->getParentOfType<ModuleOp>();
  // auxiliary builder used for create operations in shape func
  // original builder maybe a rewriter, used for create operations in specific
  // pattern.
  OpBuilder auxiliaryBuilder(moduleOp);
  StringRef funcName = callOp.getCallee();
  auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);

  // clone funcOp, newFuncOp used for deduce function shape
  std::string newFuncName = funcName.str() + "_Shape";
  auxiliaryBuilder.setInsertionPointToStart(moduleOp.getBody());
  auto newFuncOp = auxiliaryBuilder.create<func::FuncOp>(
      funcOp->getLoc(), newFuncName, funcOp.getFunctionType());
  newFuncOp.setPrivate();
  IRMapping emptyBvm;
  funcOp.cloneInto(newFuncOp, emptyBvm);

  // replace the operands of returnOp with corresponding shape
  func::ReturnOp retOp = *newFuncOp.getOps<func::ReturnOp>().begin();
  if (!retOp) {
    newFuncOp->erase();
    return failure();
  }

  for (Value &&retTensor : retOp.getOperands()) {
    auto retTy = retTensor.getType();
    if (!retTy.isa<RankedTensorType>()) {
      newFuncOp->erase();
      return failure();
    }
  }

  SmallVector<Type> allResultTypes;
  SmallVector<Value> allResults;

  auxiliaryBuilder.setInsertionPoint(retOp);
  for (Value &&retTensor : retOp.getOperands()) {
    auto retShape =
        auxiliaryBuilder.create<shape::ShapeOfOp>(retOp.getLoc(), retTensor);
    allResultTypes.emplace_back(retShape.getType());
    allResults.emplace_back(retShape);
  }

  // return the shape of original tensor returned by function
  auto newRetOp =
      auxiliaryBuilder.create<func::ReturnOp>(retOp.getLoc(), allResults);
  auto newFuncType = auxiliaryBuilder.getFunctionType(
      newFuncOp.getArgumentTypes(), allResultTypes);
  newFuncOp.setFunctionType(newFuncType);
  retOp->erase();

  // reify newFunc to get the shape computation for current callOp
  {
    PassManager pm(moduleOp->getContext(), func::FuncOp::getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createByteIRShapeReificationPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (mlir::failed(pm.run(newFuncOp))) {
      newFuncOp->erase();
      return failure();
    }
  }

  // collect all shape computation ops
  SmallVector<Operation *> reificationOps = collectAllOpsForReturn(newRetOp);

  // value only depends on the shape of FuncArgs.
  for (Value &&ret : newRetOp.getOperands()) {
    if (!deduceFromFuncArgShape(ret)) {
      newFuncOp->erase();
      return failure();
    }
  }

  // mapping the shape computation ops and collect reifications
  {
    mlir::computeTopologicalSorting(reificationOps);

    IRMapping bvm;
    size_t numArg = newFuncOp.getNumArguments();
    for (size_t i = 0; i < numArg; ++i) {
      bvm.map(newFuncOp.getArgument(i), callOp.getOperand(i));
    }

    builder.setInsertionPoint(callOp);

    for (Operation *oldOp : reificationOps) {
      auto newOp = builder.clone(*oldOp, bvm);
    }

    for (Value &&ret : newRetOp.getOperands()) {
      reifications.push_back(bvm.lookup(ret));
    }
  }

  // remove newFuncOp
  newFuncOp->erase();
  return success();
}

} // namespace

LogicalResult mlir::reifyShapes(OpBuilder &builder, Operation *op,
                                SmallVectorImpl<Value> &reifications) {
  if (!op)
    return failure();

  if (op->hasTrait<hlo::OpTrait::CompatibleOperandsAndResultType>()) {
    // CompatibleOperandsAndResultType does not implement reify
    reifications.push_back(
        builder.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0)));
    return success();
  }

  // TODO: support nested function call
  if (auto origin = dyn_cast<InferShapedTypeOpInterface>(op)) {
    if (failed(origin.reifyReturnTypeShapes(builder, origin->getOperands(),
                                            reifications))) {
      return failure();
    }
  } else if (auto reifyFunc =
                 reifyReturnTypeShapes(op->getName().getStringRef())) {
    if (failed(reifyFunc(op, builder, op->getOperands(), reifications))) {
      return failure();
    }
  } else if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    auto inferFunc = reifyReturnTypeShapes(customCall.getCallTargetName());
    if (!inferFunc) {
      return failure();
    }
    if (failed(inferFunc(op, builder, op->getOperands(), reifications)))
      return failure();
  } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
    if (failed(reifyCallOp(builder, op, reifications))) {
      return failure();
    }
  } else if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (OpResult &&result : op->getOpResults()) {
      auto tiedOperand = dpsOp.getTiedOpOperand(result);
      reifications.push_back(
          builder.create<shape::ShapeOfOp>(op->getLoc(), tiedOperand->get()));
    }
  } else {
    // Return failure if op doesn't have InferShapedTypeOpInterface and not
    // registered.
    return failure();
  }

  return success();
}
