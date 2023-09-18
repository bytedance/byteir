//===- ShardingInterface.cpp -------------------------------------*- C++-*-===//
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

#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sharding-interface"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.cpp.inc"

// ---------------------------------------------------------------------------
//
// common util functions
//
// ---------------------------------------------------------------------------
namespace {

static FailureOr<ShardingOption> getShardingOptionFromAttr(Operation *op) {
  if (auto shardingAttr = op->getAttrOfType<ArrayAttr>(getShardingAttrName())) {
    FailureOr<SmallVector<SmallVector<int64_t>>> shardingOptionFromAttr =
        getArrayOfIntArray(shardingAttr);
    return shardingOptionFromAttr;
  }
  return failure();
}

} // namespace

// TODO: handle multiple operand annotations
FailureOr<SmallVector<SmallVector<int64_t>>>
mesh::getShardingAnnotation(OpResult result, bool mergeOperandAnnotations) {
  Value val = result.cast<Value>();
  bool isAnnotated = llvm::any_of(val.getUsers(), [&](Operation *user) {
    auto annotateOp = llvm::dyn_cast<mesh::AnnotateOp>(user);
    if (!annotateOp)
      return false;
    if (mergeOperandAnnotations)
      return true;
    return annotateOp.getAsResult();
  });
  if (isAnnotated) {
    assert(userCount(val) == 1);
    auto annotateOp = llvm::cast<mesh::AnnotateOp>(*val.getUsers().begin());
    if (!annotateOp.getAsResult()) {
      annotateOp.setAsResult(true);
    }
    ArrayAttr tensorSharding = annotateOp.getSharding();
    FailureOr<SmallVector<SmallVector<int64_t>>> shardingArray =
        getArrayOfIntArray(tensorSharding);
    assert(succeeded(shardingArray));
    return shardingArray;
  }
  return failure();
}

// ---------------------------------------------------------------------------
//
// ShardingInterface::verifyShardingInterfaceImpl
//
// ---------------------------------------------------------------------------

LogicalResult mesh::ShardingInterface::verifyShardingInterfaceImpl() {
  Operation *op = getOperation();

  // check loop types
  SmallVector<ShardingIteratorType> loopTypes = getLoopIteratorTypes();
  if (loopTypes.size() == 0)
    return failure();

  // check maps
  SmallVector<AffineMap> maps = getIndexingMaps();
  if (maps.size() == 0)
    return failure();
  unsigned numOperands = op->getNumOperands();
  unsigned numResults = op->getNumResults();
  if (numOperands + numResults != maps.size())
    return failure();

  for (OpResult result : op->getResults()) {
    auto resultType = result.getType().dyn_cast<RankedTensorType>();
    if (!resultType)
      return failure();
    AffineMap map = maps[numOperands + result.getResultNumber()];
    if (!map.isProjectedPermutation()) {
      return failure();
    }
  }

  return success();
}

// ---------------------------------------------------------------------------
//
// ShardingInterface::getShardingOption
//
// ---------------------------------------------------------------------------

// Default implementation logic:
// 1. Check for Existing Attribute: If the operation already possesses a
//     `ShardingOption`` attribute, return this attribute immediately.
// 2. Initialization: Instantiate an empty `ShardingOption``. This should be an
//     array containing int64 sub-arrays, each corresponding to a loop in the
//     operation.
// 3. Results Annotation Handling:
//     - Iterate over all the results of the operation, If a result has an
//       annotation:
//       - Map the tensor dimensions to loop iterators.
//       - Set the corresponding axes based on the mapped loop iterators.
//       - In cases where there's a conflict with previously set axes, it
//       implies
//         an invalid sharding annotation. In such instances, flag this
//         inconsistency for subsequent error handling or correction.
// 4. Operands Annotation Handling:
//     - Iterate over all the operands of the operation, using the information
//       from:
//       - Reduction iterator loops and
//       - Unhandled parallel iterator loops
//     - Validate the remaining iterator loops. If discrepancies arise during
//       validation, take appropriate corrective actions or raise errors.
// 5. Replication of Mesh Axes: Any mesh axes that haven't been addressed or
//     mapped during the above steps should be treated as replicated axes.
// 6. Return Logic:
//     - If the constructed or modified ShardingOption is valid, return it.
//     - If inconsistencies or errors were detected, return a `failure()``.
FailureOr<ShardingOption> mesh::detail::defaultGetShardingOption(Operation *op,
                                                                 OpBuilder &b) {
  ShardingInterface shardingOp = llvm::cast<ShardingInterface>(op);
  SmallVector<SmallVector<int64_t>> shardingOption;

  // 1. if there is a valid sharding attr, use it.
  bool existShardingOptionAttr = false;
  FailureOr<SmallVector<SmallVector<int64_t>>> shardingOptionFromAttr =
      getShardingOptionFromAttr(op);
  if (succeeded(shardingOptionFromAttr)) {
    existShardingOptionAttr = true;
    shardingOption = *shardingOptionFromAttr;
  }

  // 2. infer sharding option from results
  if (!existShardingOptionAttr) {
    if (failed(shardingOp.verifyShardingInterfaceImpl()))
      return failure();
    SmallVector<ShardingIteratorType> loopTypes =
        shardingOp.getLoopIteratorTypes();
    SmallVector<AffineMap> maps = shardingOp.getIndexingMaps();
    unsigned numOperands = op->getNumOperands();

    shardingOption.resize(loopTypes.size());
    for (OpResult result : op->getResults()) {
      auto resultType = result.getType().dyn_cast<RankedTensorType>();
      AffineMap map = maps[numOperands + result.getResultNumber()];
      FailureOr<SmallVector<SmallVector<int64_t>>> shardingArray =
          getShardingAnnotation(result, true);
      if (succeeded(shardingArray)) {
        for (auto it : llvm::zip(map.getResults(), *shardingArray)) {
          AffineExpr expr = std::get<0>(it);
          const SmallVector<int64_t> &axis = std::get<1>(it);
          auto dim = expr.cast<AffineDimExpr>();
          unsigned position = dim.getPosition();
          // TODO: figure out a valid sharding option even if some of the tensor
          // sharding annotation don't match
          if (!shardingOption[position].empty() &&
              shardingOption[position] != axis)
            return failure();
          else
            shardingOption[position].append(axis);
        }

        if (int64_t(shardingArray->size()) > resultType.getRank()) {
          assert(int64_t(shardingArray->size()) == resultType.getRank() + 1);
          bool findReduce = false;
          SmallVector<int64_t> axis =
              (*shardingArray)[shardingArray->size() - 1];
          for (auto it : llvm::enumerate(loopTypes)) {
            int64_t position = it.index();
            ShardingIteratorType loopType = it.value();
            if (loopType == ShardingIteratorType::reduction_sum) {
              if (!shardingOption[position].empty() &&
                  shardingOption[position] != axis)
                continue;
              if (shardingOption[position].empty())
                shardingOption[position].append(axis);
              findReduce = true;
              break;
            }
          }
          if (!findReduce)
            return failure();
        }
      }
    }
  }

  // 3. infer sharding option from results
  // TODO

  // 4. set sharding option attr
  simplifyShardingOptionOrAnnotation(shardingOption);
  if (!shardingOption.empty()) {
    ArrayAttr shardingOptionAttr =
        convertArrayOfI64ArrayToAttr(b, shardingOption);
    op->setAttr(getShardingAttrName(), shardingOptionAttr);
  }

  return shardingOption;
}

// ---------------------------------------------------------------------------
//
// ShardingInterface::setShardingAnnotations
//
// ---------------------------------------------------------------------------

namespace {

static LogicalResult
checkOperandAffineExprRecursively(AffineExpr expr,
                                  SmallVectorImpl<bool> &seenIds) {
  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = binOpExpr.getLHS();
    AffineExpr rhs = binOpExpr.getRHS();
    if (failed(checkOperandAffineExprRecursively(lhs, seenIds)))
      return failure();
    if (failed(checkOperandAffineExprRecursively(rhs, seenIds)))
      return failure();
    return success();
  }
  case AffineExprKind::Mul: {
    auto binOpExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = binOpExpr.getLHS();
    AffineExpr rhs = binOpExpr.getRHS();
    AffineExpr dimExpr;
    if (lhs.getKind() == AffineExprKind::DimId) {
      dimExpr = lhs;
      if (rhs.getKind() != AffineExprKind::Constant)
        return failure();
    } else if (rhs.getKind() == AffineExprKind::DimId &&
               lhs.getKind() == AffineExprKind::Constant) {
      dimExpr = rhs;
    } else
      return failure();
    unsigned position = dimExpr.cast<AffineDimExpr>().getPosition();
    if ((size_t)position >= seenIds.size() || seenIds[position])
      return failure();
    seenIds[position] = true;
    return success();
  }
  case AffineExprKind::DimId: {
    unsigned position = expr.cast<AffineDimExpr>().getPosition();
    if ((size_t)position >= seenIds.size() || seenIds[position])
      return failure();
    seenIds[position] = true;
    return success();
  }
  default:
    return failure();
  }
}

static FailureOr<DenseSet<unsigned>> checkOperandAffineExpr(AffineExpr expr,
                                                            unsigned numDims) {
  SmallVector<bool> seenIds(numDims, false);
  if (failed(checkOperandAffineExprRecursively(expr, seenIds)))
    return failure();

  DenseSet<unsigned> positions;
  for (auto it : llvm::enumerate(seenIds)) {
    if (it.value())
      positions.insert((unsigned)it.index());
  }
  return positions;
}

} // namespace

LogicalResult
mesh::setShardingAnnotation(OpBuilder &b, OpResult result,
                            ShardingOptionRef shardingOption, AffineMap map,
                            ArrayRef<ShardingIteratorType> loopTypes) {
  auto resultType = result.getType().dyn_cast<RankedTensorType>();
  SmallVector<SmallVector<int64_t>> shardingAnnotation(resultType.getRank() +
                                                       1);
  for (auto it : llvm::enumerate(map.getResults())) {
    AffineExpr expr = it.value();
    auto dim = expr.cast<AffineDimExpr>();
    unsigned position = dim.getPosition();
    if (position < shardingOption.size())
      shardingAnnotation[it.index()].append(shardingOption[position]);
  }

  SmallVector<int64_t> reduceSumAxis;
  for (auto it : llvm::zip(loopTypes, shardingOption)) {
    ShardingIteratorType loopType = std::get<0>(it);
    if (loopType == ShardingIteratorType::reduction_sum) {
      const SmallVector<int64_t> &axis = std::get<1>(it);
      reduceSumAxis.append(axis);
    }
  }
  shardingAnnotation[shardingAnnotation.size() - 1].append(reduceSumAxis);
  simplifyShardingOptionOrAnnotation(shardingAnnotation);
  ArrayAttr shardingAnnotationAttr =
      convertArrayOfI64ArrayToAttr(b, shardingAnnotation);

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(result);
  auto annotateOp =
      b.create<mesh::AnnotateOp>(result.getLoc(), resultType, result,
                                 shardingAnnotationAttr, /*required*/ false);
  result.replaceAllUsesExcept(annotateOp, annotateOp);
  return success();
}

LogicalResult
mesh::setShardingAnnotation(OpBuilder &b, OpOperand &opOperand,
                            ShardingOptionRef shardingOption, AffineMap map,
                            ArrayRef<ShardingIteratorType> loopTypes) {
  Value operand = opOperand.get();
  auto defOp = operand.getDefiningOp<AnnotateOp>();
  if (defOp && !defOp.getAsResult()) {
    return success();
  }

  auto type = operand.getType().dyn_cast<RankedTensorType>();
  SmallVector<SmallVector<int64_t>> shardingAnnotation(type.getRank() + 1);
  unsigned numDims = map.getNumDims();
  for (auto it : llvm::enumerate(map.getResults())) {
    int64_t idx = it.index();
    AffineExpr expr = it.value();
    FailureOr<DenseSet<unsigned>> positions =
        checkOperandAffineExpr(expr, numDims);
    if (failed(positions))
      return failure();
    SmallVector<unsigned> shardedPotisions;
    for (unsigned position : *positions) {
      if ((size_t)position < shardingOption.size() &&
          !shardingOption[position].empty())
        shardedPotisions.push_back(position);
    }
    // mostly one sharded position is accepted
    if (shardedPotisions.size() > 1)
      return failure();
    if (shardedPotisions.size() == 1) {
      shardingAnnotation[idx].append(shardingOption[shardedPotisions[0]]);
    }
  }

  simplifyShardingOptionOrAnnotation(shardingAnnotation);
  ArrayAttr shardingAnnotationAttr =
      convertArrayOfI64ArrayToAttr(b, shardingAnnotation);

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(operand);
  auto annotateOp = b.create<mesh::AnnotateOp>(
      operand.getLoc(), type, operand, shardingAnnotationAttr,
      /*required*/ false, /*as_result*/ false);
  operand.replaceAllUsesExcept(annotateOp, annotateOp);

  return success();
}

// Default implementation logic:
// 1. Results Annotation Handling: Given the constraints of the result indexing
//     maps, which are limited to projected permutations, there can only be a
//     single DimId across all the result indexing maps.
//   - For parallel loop iterators: Establish and assign the corresponding axes
//     based on the mapped loop iterators.
//   - For reduction loops: Append additional axes to the end of the existing
//     annotations to indicate their association with the reduction loops.
// 2. Operands Annotation Handling: Operand annotations pose a more intricate
//     challenge compared to results due to the possibility that they might not
//     strictly adhere to projected permutations.
//     - Here, we constrain the results of the operand's indexing maps to a
//       representation format: c_i * d_i + c_j * d_j + ..., In this
//       representation:
//       - c_i and c_j denote constants. If a constant has a value of one, it
//       may be excluded from the representation.
//       - ​d_i and d_j represent the `DimId`.
//     - In situations where the representation contains multiple `DimId`s:
//       Sharding can only be applied to at most one of them. This constraint
//       ensures that the operand annotations don't introduce excessive
//       complexity and retain predictability in their sharding behavior.
LogicalResult mesh::detail::defaultSetShardingAnnotations(Operation *op,
                                                          OpBuilder &b) {

  // 1. get sharding option from attribute
  ShardingInterface shardingOp = llvm::cast<ShardingInterface>(op);
  FailureOr<SmallVector<SmallVector<int64_t>>> shardingOptionFromAttr =
      getShardingOptionFromAttr(op);
  if (failed(shardingOptionFromAttr))
    return failure();

  SmallVector<ShardingIteratorType> loopTypes =
      shardingOp.getLoopIteratorTypes();
  SmallVector<AffineMap> maps = shardingOp.getIndexingMaps();
  unsigned numOperands = op->getNumOperands();

  // 2. set sharding annotations for all results
  for (OpResult result : op->getResults()) {
    if (succeeded(getShardingAnnotation(result, false)))
      continue;

    if (failed(setShardingAnnotation(
            b, result, *shardingOptionFromAttr,
            maps[numOperands + result.getResultNumber()], loopTypes)))
      return failure();
  }

  // 3. set sharding annotations for all operands
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (failed(setShardingAnnotation(b, opOperand, *shardingOptionFromAttr,
                                     maps[opOperand.getOperandNumber()],
                                     loopTypes)))
      return failure();
  }

  return success();
}

// ---------------------------------------------------------------------------
//
// ShardingInterface::printLoopTypesAndIndexingMaps
//
// ---------------------------------------------------------------------------

void mesh::ShardingInterface::printLoopTypesAndIndexingMaps(raw_ostream &os) {
  os << "print loop types and indexing maps for: \n";
  getOperation()->print(os);
  os << "\n";
  os << "loop types: [";
  for (ShardingIteratorType type : getLoopIteratorTypes()) {
    os << stringifyEnum(type) << " ";
  }
  os << "]\n";
  os << "indexing maps: \n";
  for (AffineMap map : getIndexingMaps())
    os << map << "\n";
  os << "\n";
}

// ---------------------------------------------------------------------------
//
// mesh::createCclOpBetweenShardings
//
// ---------------------------------------------------------------------------

namespace {

static SmallVector<ArrayAttr> convertOutmostToVector(ArrayAttr arrayAttr) {
  SmallVector<ArrayAttr> res;
  for (Attribute attr : arrayAttr) {
    ArrayAttr subAttr = attr.cast<ArrayAttr>();
    res.push_back(subAttr);
  }
  return res;
}

} // namespace

// Implementation Logic: At this stage, the logic for communication creation can
// be kept straightforward. Further canonicalization and optimization of these
// communications can be executed later. The process can be categorized into
// three stages:
//
// 1. All-Reduce: If any reduction sharding axes are absent in the
// current annotation operation relative to its operand's defining operation
// (which should also be an annotation operation with as_result = true), an
// all-reduce operation should be initialized.
//
// 2. All-Gather: Create an all-gather operation to reconstruct the complete
// tensor
//
// 3. Local-Split: Launch a local-split operation to derive the final sharded
// tensor.
FailureOr<Value> mlir::mesh::createCclOpBetweenShardings(
    OpBuilder &b, Value src, ArrayRef<SmallVector<int64_t>> shardingFromOrigin,
    ArrayRef<SmallVector<int64_t>> shardingToOrigin) {
  MLIRContext *ctx = b.getContext();
  auto type = src.getType().dyn_cast<RankedTensorType>();
  int64_t rank = type.getRank();

  SmallVector<SmallVector<int64_t>> shardingFrom(shardingFromOrigin);
  SmallVector<SmallVector<int64_t>> shardingTo(shardingToOrigin);
  if ((int64_t)shardingFrom.size() < rank + 1)
    shardingFrom.append(rank + 1 - shardingFrom.size(), {});
  if ((int64_t)shardingTo.size() < rank + 1)
    shardingTo.append(rank + 1 - shardingTo.size(), {});

  DenseSet<int64_t> toPartialSumAxisSet(shardingTo[rank].begin(),
                                        shardingTo[rank].end());
  DenseSet<int64_t> fromPartialSumAxisSet(shardingFrom[rank].begin(),
                                          shardingFrom[rank].end());
  if (!llvm::set_is_subset(toPartialSumAxisSet, fromPartialSumAxisSet))
    return failure();
  llvm::set_subtract(fromPartialSumAxisSet, toPartialSumAxisSet);
  SmallVector<int64_t> allReduceAxis = llvm::to_vector(fromPartialSumAxisSet);
  SmallVector<int64_t> toPartialSumAxis = shardingTo.back();
  SmallVector<int64_t> fromPartialSumAxis = shardingFrom.back();
  shardingFrom.pop_back();
  shardingTo.pop_back();

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(src);
  Value curVal = src;
  RankedTensorType srcType = src.getType().cast<RankedTensorType>();

  if (!allReduceAxis.empty()) {
    SmallVector<SmallVector<int64_t>> arrayOfArray = shardingFrom;
    arrayOfArray.push_back(toPartialSumAxis);
    simplifyShardingOptionOrAnnotation(arrayOfArray);
    mesh::MeshShardingAttr shardEncoding;
    if (!arrayOfArray.empty()) {
      SmallVector<ArrayAttr> encodingArray =
          convertOutmostToVector(convertArrayOfI64ArrayToAttr(b, arrayOfArray));
      shardEncoding = MeshShardingAttr::get(ctx, encodingArray);
    }
    auto resType =
        RankedTensorType::get(srcType.getShape(), srcType.getElementType(),
                              shardEncoding ? shardEncoding : nullptr);
    ArrayAttr attr = b.getI64ArrayAttr(allReduceAxis);
    auto allReduceOp =
        b.create<mesh::AllReduceOp>(src.getLoc(), resType, curVal, attr, "sum");
    curVal = allReduceOp.getResult();
  }

  simplifyShardingOptionOrAnnotation(shardingFrom);
  simplifyShardingOptionOrAnnotation(shardingTo);
  if (!shardingFrom.empty()) {
    ArrayAttr attr = convertArrayOfI64ArrayToAttr(b, shardingFrom);
    mesh::MeshShardingAttr shardEncoding;
    if (!toPartialSumAxis.empty()) {
      SmallVector<ArrayAttr> encodingArray(rank);
      encodingArray.push_back(b.getI64ArrayAttr(toPartialSumAxis));
      shardEncoding = MeshShardingAttr::get(ctx, encodingArray);
    }
    SmallVector<int64_t> tensorAxis;
    for (auto it : llvm::enumerate(shardingFrom)) {
      if (!it.value().empty())
        tensorAxis.push_back(it.index());
    }
    auto resType =
        RankedTensorType::get(srcType.getShape(), srcType.getElementType(),
                              shardEncoding ? shardEncoding : nullptr);
    auto allGahterOp = b.create<mesh::AllGatherOp>(
        src.getLoc(), resType, curVal, attr, b.getI64ArrayAttr(tensorAxis));
    curVal = allGahterOp.getResult();
  }

  if (!shardingTo.empty()) {
    ArrayAttr localSplitAttr = convertArrayOfI64ArrayToAttr(b, shardingTo);
    SmallVector<ArrayAttr> encodingArray =
        convertOutmostToVector(localSplitAttr);
    if (!toPartialSumAxis.empty()) {
      if ((int64_t)encodingArray.size() < rank) {
        encodingArray.append(rank - encodingArray.size(),
                             b.getI64ArrayAttr({}));
      }
      encodingArray.push_back(b.getI64ArrayAttr(toPartialSumAxis));
    }
    auto shardEncoding = MeshShardingAttr::get(ctx, encodingArray);
    auto resType = RankedTensorType::get(
        srcType.getShape(), srcType.getElementType(), shardEncoding);
    auto localSplitOp = b.create<mesh::LocalSplitOp>(src.getLoc(), resType,
                                                     curVal, localSplitAttr);
    curVal = localSplitOp.getResult();
  }

  return curVal;
}
