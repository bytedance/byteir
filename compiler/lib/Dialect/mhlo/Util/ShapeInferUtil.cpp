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
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

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
  } else {
    // Return failure if op doesn't have InferShapedTypeOpInterface and not
    // registered.
    return failure();
  }

  return success();
}
