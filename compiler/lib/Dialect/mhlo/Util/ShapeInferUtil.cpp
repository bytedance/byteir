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

namespace {

using ResultShapes = SmallVector<ArrayRef<int64_t>, 1>;

LogicalResult checkAndSetTypes(Operation *op,
                               const ResultShapes &inferredShapes,
                               bool overrideOldShape = false) {
  auto results = op->getResults();

  // check
  for (auto it : llvm::zip(results, inferredShapes)) {
    auto result = std::get<0>(it);
    auto resultShape = result.getType().dyn_cast<ShapedType>();
    if (!resultShape || !resultShape.hasRank())
      continue;
    auto inferredShape = std::get<1>(it);

    if (!overrideOldShape &&
        resultShape.getRank() != int64_t(inferredShape.size())) {
      op->emitError()
          << "Found rank mismatch during shape inferring, previous is "
          << resultShape.getRank() << ", inferred is " << inferredShape.size()
          << "\n";
      return failure();
    }

    for (auto dimIt : llvm::zip(resultShape.getShape(), inferredShape)) {
      int64_t lDim = std::get<0>(dimIt);
      int64_t rDim = std::get<1>(dimIt);
      if (!overrideOldShape && lDim > 0 && rDim > 0 && lDim != rDim) {
        op->emitError()
            << "Found dimension mismatch during shape inferring, previous is "
            << lDim << ", inferred is " << rDim << "\n";
        return failure();
      }
    }
  }

  // set
  for (auto it : llvm::zip(results, inferredShapes)) {
    Value result = std::get<0>(it);
    auto shapedType = result.getType().dyn_cast<ShapedType>();
    if (!shapedType)
      continue;
    result.setType(shapedType.clone(std::get<1>(it)));
  }

  return success();
}

LogicalResult inferBoundedShapeUsingRegistry(Operation *op) {
  InferBoundedReturnTypeComponents inferFunc = nullptr;
  if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    inferFunc =
        inferBoundedReturnTypeComponents(customCall.getCallTargetName());
  } else {
    inferFunc = inferBoundedReturnTypeComponents(op->getName().getStringRef());
  }

  if (nullptr == inferFunc) {
    return success();
  }

  llvm::SmallVector<ShapedTypeComponents> resultShapeComponnets;
  LogicalResult inferStatus = inferFunc(
      op->getContext(), op->getLoc(), op->getOperands(),
      op->getAttrDictionary(), op->getRegions(), resultShapeComponnets);
  if (failed(inferStatus)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Registered InferBoundedReturnTypeComponents failed for "
               << *op << "\n");
    return success();
  }

  ResultShapes resultShapes = llvm::to_vector(llvm::map_range(
      resultShapeComponnets,
      [](const ShapedTypeComponents &comp) { return comp.getDims(); }));
  return checkAndSetTypes(op, resultShapes);
}

LogicalResult
inferShapeUsingSameOperandsAndResultShapeTrait(Operation *op,
                                               bool overrideOldShape) {
  Type staticShapedType = nullptr;
  for (Type t : op->getOperandTypes()) {
    if (auto shape_type = t.dyn_cast<ShapedType>()) {
      if (shape_type.hasStaticShape()) {
        staticShapedType = t;
        break;
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Operand type " << t << " is not ShapedType in op: " << *op
                 << ", skip infer\n");
      return success();
    }
  }
  if (!staticShapedType) {
    LLVM_DEBUG(llvm::dbgs() << "There's no operand type with static shape in "
                            << *op << "\n");
    return success();
  }

  ResultShapes resultShapes(op->getNumResults(),
                            staticShapedType.cast<ShapedType>().getShape());
  return checkAndSetTypes(op, resultShapes, overrideOldShape);
}

LogicalResult
inferShapeUsingInferShapedTypeOpInterface(InferShapedTypeOpInterface op,
                                          bool overrideOldShape) {
  SmallVector<ShapedTypeComponents> resultShapeComps;
  SmallVector<Value> operands = llvm::to_vector(op->getOperands());
  ValueShapeRange operandsShapeRange(operands);
  LogicalResult inferStatus = op.inferReturnTypeComponents(
      op->getContext(), op->getLoc(), operandsShapeRange,
      op->getAttrDictionary(), op->getRegions(), resultShapeComps);
  if (failed(inferStatus)) {
    LLVM_DEBUG(llvm::dbgs()
               << "InferReturnTypeComponents failed for " << op << "\n");
    return failure();
  }

  ResultShapes resultShapes = llvm::to_vector(
      llvm::map_range(resultShapeComps, [](const ShapedTypeComponents &comp) {
        return comp.getDims();
      }));

  auto ret = checkAndSetTypes(op, resultShapes, overrideOldShape);
  return ret;
}

LogicalResult inferShapeUsingInferTypeOpInterface(InferTypeOpInterface op,
                                                  bool overrideOldShape) {
  SmallVector<Type> resultShapeTypes;
  LogicalResult inferStatus = op.inferReturnTypes(
      op->getContext(), op->getLoc(), op->getOperands(),
      op->getAttrDictionary(), op->getRegions(), resultShapeTypes);
  if (failed(inferStatus)) {
    LLVM_DEBUG(llvm::dbgs() << "InferReturnTypes failed for " << op << "\n");
    return failure();
  }
  for (auto it : llvm::zip(op->getResults(), resultShapeTypes)) {
    Value result = std::get<0>(it);
    result.setType(std::get<1>(it));
  }

  ResultShapes resultShapes =
      llvm::to_vector(llvm::map_range(resultShapeTypes, [](Type t) {
        if (auto shape = t.dyn_cast<ShapedType>())
          return shape.getShape();
        return ArrayRef<int64_t>(std::nullopt);
      }));
  return checkAndSetTypes(op, resultShapes, overrideOldShape);
}

bool isShape(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<shape::ShapeDialect>(dialect);
}

bool isTensor(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<tensor::TensorDialect>(dialect);
}

bool isArith(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && isa<arith::ArithDialect>(dialect);
}

LogicalResult inferResultShapes(Operation *op, bool isBoundedShapeInfer,
                                DenseMap<Value, Attribute> &foldMap,
                                bool overrideOldShape) {

  if (isBoundedShapeInfer && (isShape(op) || isTensor(op) || isArith(op))) {
    // skip the inference in shape dialect and tensor dialect
    SmallVector<Attribute, 4> operands;
    SmallVector<OpFoldResult, 4> foldResults;
    for (auto value : op->getOperands()) {
      if (!foldMap.count(value)) {
        operands.push_back({});
      } else {
        operands.push_back(foldMap[value]);
      }
    }
    if (op->fold(operands, foldResults).failed()) {
      return success();
    }
    for (auto value : llvm::zip(op->getResults(), foldResults)) {
      auto resultValue = std::get<0>(value);
      auto foldResult = std::get<1>(value);
      // set the fold result to tensor encoding
      if (foldResult.dyn_cast<Attribute>()) {
        foldMap[resultValue] = foldResult.dyn_cast<Attribute>();
        if (auto ty = resultValue.getType().dyn_cast<RankedTensorType>()) {
          auto newType = RankedTensorType::get(
              ty.getShape(), ty.getElementType(),
              DictionaryAttr::get(
                  op->getContext(),
                  {NamedAttribute(
                      StringAttr::get(op->getContext(),
                                      getBoundedShapeDenseAttrName()),
                      foldResult.dyn_cast<Attribute>())}));
          resultValue.setType(newType);
        }
      }
    }
  } else if (isBoundedShapeInfer &&
             failed(inferBoundedShapeUsingRegistry(op))) {
    return failure();
  } else if (op->hasTrait<OpTrait::SameOperandsAndResultShape>() &&
             failed(inferShapeUsingSameOperandsAndResultShapeTrait(
                 op, overrideOldShape))) {
    // Note: some ops has InferShapedTypeOpInterface but it will return
    // failure() directly in the implementation, therefore
    // SameOperandsAndResultShape trait should be checked before checking
    // InferShapedTypeOpInterface
    return failure();
  } else if (dyn_cast<InferShapedTypeOpInterface>(op) &&
             failed(inferShapeUsingInferShapedTypeOpInterface(
                 dyn_cast<InferShapedTypeOpInterface>(op), overrideOldShape))) {
    return failure();
  } else if (dyn_cast<InferTypeOpInterface>(op) &&
             failed(inferShapeUsingInferTypeOpInterface(
                 dyn_cast<InferTypeOpInterface>(op), overrideOldShape))) {
    return failure();
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// runShapeInference
//===----------------------------------------------------------------------===//

// TODO: supported nested function call
LogicalResult mlir::runShapeInference(func::FuncOp funcOp,
                                      bool isBoundedShapeInfer,
                                      bool overrideOldShape) {
  DenseMap<Value, Attribute> foldMap;
  bool interrupted =
      funcOp
          ->walk([&](Operation *op) {
            if (failed(inferResultShapes(op, isBoundedShapeInfer, foldMap,
                                         overrideOldShape))) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted();

  if (interrupted)
    return failure();

  func::ReturnOp retOp = *funcOp.getOps<func::ReturnOp>().begin();
  funcOp.setType(FunctionType::get(
      funcOp.getContext(), funcOp.getArgumentTypes(), retOp.getOperandTypes()));

  return success();
}

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
