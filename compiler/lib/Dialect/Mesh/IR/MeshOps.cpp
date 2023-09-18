//===- MeshOps.cpp --------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "mesh-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

#include "byteir/Dialect/Mesh/IR/MeshOpsDialect.cpp.inc"

namespace {

FailureOr<SmallVector<int64_t>> getRidIfBeginWith(ArrayRef<int64_t> array,
                                                  ArrayRef<int64_t> begining) {
  if (begining.size() > array.size())
    return failure();
  for (size_t i = 0; i < begining.size(); ++i) {
    if (begining[i] != array[i])
      return failure();
  }
  return SmallVector<int64_t>{array.drop_front(begining.size())};
}

FailureOr<SmallVector<int64_t>> getRedIfEndWith(ArrayRef<int64_t> array,
                                                ArrayRef<int64_t> ending) {
  if (ending.size() > array.size())
    return failure();

  int64_t sizeOffset = array.size() - ending.size();
  for (int64_t i = (int64_t)ending.size() - 1; i >= 0; i--) {
    if (ending[i] != array[i + sizeOffset])
      return failure();
  }
  return SmallVector<int64_t>{array.drop_back(ending.size())};
}

} // namespace

//===----------------------------------------------------------------------===//
// common util function
//===----------------------------------------------------------------------===//

void mlir::mesh::simplifyShardingOptionOrAnnotation(
    SmallVector<SmallVector<int64_t>> &array) {
  for (int64_t i = array.size() - 1; i >= 0; i--) {
    if (array[i].empty())
      array.pop_back();
    else
      break;
  }
}

void mlir::mesh::populateMeshOpsCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  LocalSplitOp::getCanonicalizationPatterns(patterns, ctx);
}

FailureOr<ClusterOp> mlir::mesh::getMeshClusterOp(func::FuncOp funcOp) {
  SymbolRefAttr meshClusterSymbol = funcOp->getAttr(getMeshClusterAttrName())
                                        .dyn_cast_or_null<SymbolRefAttr>();
  if (!meshClusterSymbol)
    return failure();

  auto clusterOp = SymbolTable::lookupNearestSymbolFrom<ClusterOp>(
      funcOp, meshClusterSymbol);

  if (!clusterOp)
    return failure();

  return clusterOp;
}

FailureOr<ClusterOp> mlir::mesh::getMeshClusterOp(Operation *op) {
  func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
  return getMeshClusterOp(funcOp);
}

namespace {

SmallVector<int64_t> getIndices(int64_t flatternIdx, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> indices;
  for (int64_t dimSize : shape) {
    auto curIdx = flatternIdx % dimSize;
    flatternIdx /= dimSize;
    indices.push_back(curIdx);
  }
  return indices;
}

int64_t getGroupIdx(ArrayRef<int64_t> indices, ArrayRef<int64_t> otherStrides,
                    ArrayRef<int64_t> otherAxes) {
  int64_t groupIdx = 0;
  for (int64_t axis : otherAxes) {
    groupIdx += indices[axis] * otherStrides[axis];
  }
  return groupIdx;
}

} // namespace

FailureOr<DenseIntElementsAttr>
mlir::mesh::getReplicaGroups(MLIRContext *ctx, ArrayRef<int64_t> clusterShape,
                             ArrayRef<int64_t> axes) {
  DenseSet<int64_t> axesSet(axes.begin(), axes.end());
  return getReplicaGroups(ctx, clusterShape, axesSet);
}

FailureOr<DenseIntElementsAttr>
mlir::mesh::getReplicaGroups(MLIRContext *ctx, ArrayRef<int64_t> clusterShape,
                             DenseSet<int64_t> axes) {
  if (llvm::any_of(clusterShape, [](int64_t dimSize) { return dimSize <= 0; }))
    return failure();
  int64_t clusterRank = (int64_t)clusterShape.size();
  if (llvm::any_of(
          axes, [&](int64_t axis) { return axis >= clusterRank || axis < 0; }))
    return failure();

  SmallVector<int64_t> otherAxes;
  for (size_t axis = 0; axis < clusterShape.size(); ++axis) {
    if (!axes.contains(axis))
      otherAxes.push_back(axis);
  }

  int64_t numGroups = 1;
  for (int64_t dim : otherAxes) {
    numGroups *= clusterShape[dim];
  }
  int64_t numDevices = 1;
  for (int64_t dimSize : clusterShape)
    numDevices *= dimSize;
  int64_t numDevicesEachGroup = numDevices / numGroups;
  SmallVector<int64_t> strides(clusterRank, 1);
  for (int i = 1; i < clusterRank; ++i) {
    strides[i] = strides[i - 1] * clusterShape[i - 1];
  }
  SmallVector<int64_t> otherStrides;
  for (int64_t dim : otherAxes)
    otherStrides.push_back(strides[dim]);

  SmallVector<SmallVector<int64_t>> replicaGroups;
  replicaGroups.resize(numGroups);

  for (int64_t deviceId = 0; deviceId < numDevices; ++deviceId) {
    SmallVector<int64_t> indices = getIndices(deviceId, clusterShape);
    int64_t groupIdx = getGroupIdx(indices, otherStrides, otherAxes);
    replicaGroups[groupIdx].push_back(deviceId);
  }
  SmallVector<int64_t> flatternReplicaGroups;
  for (const SmallVector<int64_t> &group : replicaGroups)
    flatternReplicaGroups.append(group);

  auto type = RankedTensorType::get({numGroups, numDevicesEachGroup},
                                    IntegerType::get(ctx, 64));
  return DenseIntElementsAttr::get(type, flatternReplicaGroups);
}

//===----------------------------------------------------------------------===//
// mesh dialect.
//===----------------------------------------------------------------------===//

void MeshDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Mesh/IR/MeshOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "byteir/Dialect/Mesh/IR/MeshOpsAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "byteir/Dialect/Mesh/IR/MeshOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "byteir/Dialect/Mesh/IR/MeshOpsAttributes.cpp.inc"

#include "byteir/Dialect/Mesh/IR/MeshOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// mesh.local_split
//===----------------------------------------------------------------------===//

LogicalResult
LocalSplitOp::inferReturnTypes(MLIRContext *context, std::optional<Location>,
                               ValueRange operands, DictionaryAttr attrs,
                               OpaqueProperties, RegionRange,
                               SmallVectorImpl<Type> &inferredReturnTypes) {
  Value src = operands[0];
  auto srcType = src.getType().cast<RankedTensorType>();
  Attribute encoding = srcType.getEncoding();
  MeshShardingAttr shardEncoding =
      encoding.dyn_cast_or_null<MeshShardingAttr>();
  if (!shardEncoding)
    shardEncoding = MeshShardingAttr::get(context, {});
  SmallVector<SmallVector<int64_t>> srcAxes;
  for (ArrayAttr axis : shardEncoding.getAxes()) {
    srcAxes.push_back(getI64Array(axis));
  }
  int64_t rank = srcType.getRank();
  if ((int64_t)srcAxes.size() < rank + 1) {
    srcAxes.append(rank + 1 - srcAxes.size(), {});
  }

  OperationName name = OperationName(getOperationName(), context);
  StringAttr splitAttrName = getShardingAttrName(name);
  ArrayAttr splitAttr = attrs.get(splitAttrName).cast<ArrayAttr>();
  SmallVector<SmallVector<int64_t>> splitAxes = *getArrayOfIntArray(splitAttr);

  for (size_t i = 0; i < splitAxes.size(); ++i) {
    srcAxes[i].append(splitAxes[i]);
  }
  simplifyShardingOptionOrAnnotation(srcAxes);
  SmallVector<ArrayAttr> resShardEncoding;
  OpBuilder b(context);
  for (const SmallVector<int64_t> &array : srcAxes) {
    resShardEncoding.push_back(b.getI64ArrayAttr(array));
  }
  auto resultMeshShardingAttr =
      MeshShardingAttr::get(context, resShardEncoding);
  auto resultType = RankedTensorType::get(
      srcType.getShape(), srcType.getElementType(), resultMeshShardingAttr);
  inferredReturnTypes.push_back(resultType);

  return success();
}

namespace {

struct FoldLocalSplitToAllGather : public OpRewritePattern<LocalSplitOp> {
  using OpRewritePattern<LocalSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalSplitOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Operation *defOp = src.getDefiningOp();
    MLIRContext *ctx = rewriter.getContext();

    auto allGatherOp = llvm::dyn_cast_or_null<mesh::AllGatherOp>(defOp);
    if (!allGatherOp)
      return rewriter.notifyMatchFailure(op, "src is not a mesh.all_gather.");

    ArrayAttr gatherMeshAxis = allGatherOp.getMeshAxis();
    ArrayAttr splitMeshAxis = op.getSharding();
    SmallVector<SmallVector<int64_t>> gatherArray =
        *getArrayOfIntArray(gatherMeshAxis);
    SmallVector<SmallVector<int64_t>> splitArray =
        *getArrayOfIntArray(splitMeshAxis);

    SmallVector<SmallVector<int64_t>> newGatherArray;
    for (auto it : llvm::zip(gatherArray, splitArray)) {
      FailureOr<SmallVector<int64_t>> maybeNewGather =
          getRidIfBeginWith(std::get<0>(it), std::get<1>(it));
      if (failed(maybeNewGather))
        return failure();
      newGatherArray.push_back(*maybeNewGather);
    }
    simplifyShardingOptionOrAnnotation(newGatherArray);
    ArrayAttr newGatherMeshAxis =
        convertArrayOfI64ArrayToAttr(rewriter, newGatherArray);
    DictionaryAttr gatherDictAttr =
        AllGatherOp::getDictAttr(ctx, newGatherMeshAxis);
    SmallVector<Type> gatherResultTypes;
    if (failed(AllGatherOp::inferReturnTypes(
            ctx, std::nullopt, allGatherOp.getSrc(), gatherDictAttr, nullptr,
            {}, gatherResultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "fail to infer result dtensor type "
                                         "for the generated all-gather op.");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto newAllGather = rewriter.create<AllGatherOp>(
        op.getLoc(), gatherResultTypes[0], allGatherOp.getSrc(),
        newGatherMeshAxis, allGatherOp.getTensorAxis());
    rewriter.replaceAllUsesWith(op.getResult(), newAllGather.getResult());

    return success();
  }
};

struct FoldLocalSplitFromAllGather : public OpRewritePattern<LocalSplitOp> {
  using OpRewritePattern<LocalSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalSplitOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Operation *defOp = src.getDefiningOp();
    MLIRContext *ctx = rewriter.getContext();

    auto allGatherOp = llvm::dyn_cast_or_null<mesh::AllGatherOp>(defOp);
    if (!allGatherOp)
      return rewriter.notifyMatchFailure(op, "src is not a mesh.all_gather.");

    ArrayAttr gatherMeshAxis = allGatherOp.getMeshAxis();
    ArrayAttr splitMeshAxis = op.getSharding();
    SmallVector<SmallVector<int64_t>> gatherArray =
        *getArrayOfIntArray(gatherMeshAxis);
    SmallVector<SmallVector<int64_t>> splitArray =
        *getArrayOfIntArray(splitMeshAxis);

    int64_t rank = op.getResult().getType().cast<RankedTensorType>().getRank();
    expandArrayOfArray(gatherArray, rank);
    expandArrayOfArray(splitArray, rank);

    SmallVector<SmallVector<int64_t>> newSplitArray;
    for (auto it : llvm::zip(gatherArray, splitArray)) {
      FailureOr<SmallVector<int64_t>> maybeNewSplit =
          getRidIfBeginWith(std::get<1>(it), std::get<0>(it));
      if (failed(maybeNewSplit))
        return failure();
      newSplitArray.push_back(*maybeNewSplit);
    }
    simplifyShardingOptionOrAnnotation(newSplitArray);

    // create mesh.local_split op
    OperationName lsName = OperationName(LocalSplitOp::getOperationName(), ctx);
    SmallVector<NamedAttribute> localSplitAttrs;
    localSplitAttrs.push_back(
        NamedAttribute(LocalSplitOp::getShardingAttrName(lsName),
                       convertArrayOfI64ArrayToAttr(rewriter, newSplitArray)));
    auto localSplitDictAttr = rewriter.getDictionaryAttr(localSplitAttrs);
    SmallVector<Type> localSplitResultTypes;
    if (failed(LocalSplitOp::inferReturnTypes(
            ctx, std::nullopt, allGatherOp.getSrc(), localSplitDictAttr,
            nullptr, {}, localSplitResultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "fail to infer result dtensor type "
                                         "for the generated local split op.");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto localSplit =
        rewriter.create<LocalSplitOp>(op.getLoc(), localSplitResultTypes[0],
                                      allGatherOp.getSrc(), localSplitAttrs);

    rewriter.replaceAllUsesWith(op.getResult(), localSplit.getResult());
    return success();
  }
};

struct FoldLocalSplitWithAllReduce : public OpRewritePattern<LocalSplitOp> {
  using OpRewritePattern<LocalSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalSplitOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Operation *defOp = src.getDefiningOp();
    MLIRContext *ctx = rewriter.getContext();

    auto allReduceOp = llvm::dyn_cast_or_null<mesh::AllReduceOp>(defOp);
    if (!allReduceOp)
      return rewriter.notifyMatchFailure(op, "src is not a mesh.all_reduce.");

    ArrayAttr reduceSharding = allReduceOp.getMeshAxis();
    SmallVector<Attribute> newSplitAttrs;
    int64_t reduceScatterAxis = -1;
    for (auto it : llvm::enumerate(op.getSharding())) {
      auto subAttr = it.value();
      ArrayAttr subArrayAttr = subAttr.cast<ArrayAttr>();
      if (subArrayAttr == reduceSharding) {
        reduceScatterAxis = it.index();
        newSplitAttrs.push_back(rewriter.getI64ArrayAttr({}));
      } else
        newSplitAttrs.push_back(subAttr);
    }

    if (reduceScatterAxis == -1)
      return rewriter.notifyMatchFailure(op, "fail to get matched axis.");

    // create mesh.reduce_scatter op
    OperationName rsName =
        OperationName(ReduceScatterOp::getOperationName(), ctx);
    SmallVector<NamedAttribute> reduceScatterAttrs;
    reduceScatterAttrs.push_back(
        NamedAttribute(ReduceScatterOp::getTensorAxisAttrName(rsName),
                       rewriter.getI64IntegerAttr(reduceScatterAxis)));
    reduceScatterAttrs.push_back(
        NamedAttribute(ReduceScatterOp::getReductionAttrName(rsName),
                       allReduceOp.getReductionAttr()));
    reduceScatterAttrs.push_back(
        NamedAttribute(ReduceScatterOp::getMeshAxisAttrName(rsName),
                       allReduceOp.getMeshAxisAttr()));
    auto reduceScatterDictAttr = rewriter.getDictionaryAttr(reduceScatterAttrs);

    SmallVector<Type> reduceScatterResultTypes;
    if (failed(ReduceScatterOp::inferReturnTypes(
            ctx, std::nullopt, allReduceOp.getSrc(), reduceScatterDictAttr,
            nullptr, {}, reduceScatterResultTypes)))
      return rewriter.notifyMatchFailure(
          op, "fail to infer result dtensor type "
              "for the generated reduce scatter op.");

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    auto reduceScatterOp = rewriter.create<ReduceScatterOp>(
        allReduceOp.getLoc(), reduceScatterResultTypes[0], allReduceOp.getSrc(),
        reduceScatterAttrs);

    // create mesh.local_split op
    OperationName lsName = OperationName(LocalSplitOp::getOperationName(), ctx);
    SmallVector<NamedAttribute> localSplitAttrs;
    localSplitAttrs.push_back(
        NamedAttribute(LocalSplitOp::getShardingAttrName(lsName),
                       rewriter.getArrayAttr(newSplitAttrs)));
    auto localSplitDictAttr = rewriter.getDictionaryAttr(localSplitAttrs);
    SmallVector<Type> localSplitResultTypes;
    if (failed(LocalSplitOp::inferReturnTypes(
            ctx, std::nullopt, reduceScatterOp.getResult(), localSplitDictAttr,
            nullptr, {}, localSplitResultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "fail to infer result dtensor type "
                                         "for the generated local split op.");
    auto localSplit = rewriter.create<LocalSplitOp>(
        allReduceOp.getLoc(), localSplitResultTypes[0],
        reduceScatterOp.getResult(), localSplitAttrs);

    rewriter.replaceAllUsesWith(op.getResult(), localSplit.getResult());
    return success();
  }
};

struct FoldTrivialLocalSplit : public OpRewritePattern<LocalSplitOp> {
  using OpRewritePattern<LocalSplitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalSplitOp op,
                                PatternRewriter &rewriter) const override {
    ArrayAttr splitAxisAttr = op.getSharding();
    FailureOr<SmallVector<SmallVector<int64_t>>> maybeSplitAxisArray =
        getArrayOfIntArray(splitAxisAttr);
    if (failed(maybeSplitAxisArray))
      return failure();
    SmallVector<SmallVector<int64_t>> splitAxisArray = *maybeSplitAxisArray;
    simplifyShardingOptionOrAnnotation(splitAxisArray);
    if (!splitAxisArray.empty())
      return failure();
    op.getResult().replaceAllUsesExcept(op.getSrc(), op);
    return success();
  }
};

} // namespace

void mesh::LocalSplitOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldLocalSplitWithAllReduce>(context);
  patterns.add<FoldTrivialLocalSplit>(context);
  patterns.add<FoldLocalSplitFromAllGather>(context);
  patterns.add<FoldLocalSplitToAllGather>(context);
}

//===----------------------------------------------------------------------===//
// mesh.reduce_scatter
//===----------------------------------------------------------------------===//

LogicalResult
ReduceScatterOp::inferReturnTypes(MLIRContext *context, std::optional<Location>,
                                  ValueRange operands, DictionaryAttr attrs,
                                  OpaqueProperties, RegionRange,
                                  SmallVectorImpl<Type> &inferredReturnTypes) {
  Value operand = operands[0];
  auto operandType = operand.getType().cast<RankedTensorType>();
  Attribute attr = operandType.getEncoding();
  OperationName name = OperationName(getOperationName(), context);

  if (!attr && !attr.isa<MeshShardingAttr>())
    return failure();
  auto shardingAttr = attr.dyn_cast<MeshShardingAttr>();
  StringAttr tensorAxisName = getTensorAxisAttrName(name);
  int64_t tensorAxis =
      attrs.get(tensorAxisName).cast<IntegerAttr>().getValue().getZExtValue();
  StringAttr meshAxisName = getMeshAxisAttrName(name);
  ArrayAttr meshAxis = attrs.get(meshAxisName).cast<ArrayAttr>();

  SmallVector<SmallVector<int64_t>> inpShardingArray;
  for (ArrayAttr arrayAttr : shardingAttr.getAxes()) {
    inpShardingArray.push_back(getI64Array(arrayAttr));
  }
  int64_t rank = operandType.getRank();
  if ((int64_t)inpShardingArray.size() < rank + 1)
    inpShardingArray.append(rank + 1 - inpShardingArray.size(), {});
  inpShardingArray[tensorAxis].append(getI64Array(meshAxis));

  SmallVector<int64_t> meshAxisArray = getI64Array(meshAxis);
  DenseSet<int64_t> meshAxisSet(meshAxisArray.begin(), meshAxisArray.end());
  SmallVector<int64_t> remainingPartialArray;
  for (int64_t axis : inpShardingArray[rank]) {
    if (!meshAxisSet.contains(axis))
      remainingPartialArray.push_back(axis);
  }
  inpShardingArray[rank] = remainingPartialArray;
  simplifyShardingOptionOrAnnotation(inpShardingArray);

  OpBuilder b(context);
  SmallVector<ArrayAttr> resultSharding;
  for (const SmallVector<int64_t> &array : inpShardingArray) {
    resultSharding.push_back(b.getI64ArrayAttr(array));
  }
  auto resultMeshShardingAttr = MeshShardingAttr::get(context, resultSharding);
  auto resultType = RankedTensorType::get(operandType.getShape(),
                                          operandType.getElementType(),
                                          resultMeshShardingAttr);
  inferredReturnTypes.push_back(resultType);

  return success();
}

//===----------------------------------------------------------------------===//
// mesh.all_gather
//===----------------------------------------------------------------------===//

DictionaryAttr AllGatherOp::getDictAttr(MLIRContext *ctx, ArrayAttr meshAxis) {

  OperationName name = OperationName(getOperationName(), ctx);
  SmallVector<NamedAttribute> namedAttrs;
  namedAttrs.push_back(NamedAttribute(getMeshAxisAttrName(name), meshAxis));
  return DictionaryAttr::get(ctx, namedAttrs);
}

DictionaryAttr
AllGatherOp::getDictAttr(MLIRContext *ctx,
                         ArrayRef<SmallVector<int64_t>> meshAxis) {
  OpBuilder b(ctx);
  ArrayAttr attr = convertArrayOfI64ArrayToAttr(b, meshAxis);
  return getDictAttr(ctx, attr);
}

LogicalResult
AllGatherOp::inferReturnTypes(MLIRContext *context, std::optional<Location>,
                              ValueRange operands, DictionaryAttr attrs,
                              OpaqueProperties, RegionRange,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  Value src = operands[0];
  auto srcType = src.getType().cast<RankedTensorType>();
  Attribute encoding = srcType.getEncoding();
  MeshShardingAttr shardEncoding =
      encoding.dyn_cast_or_null<MeshShardingAttr>();
  if (!shardEncoding)
    shardEncoding = MeshShardingAttr::get(context, {});
  SmallVector<SmallVector<int64_t>> srcAxes;
  for (ArrayAttr axis : shardEncoding.getAxes()) {
    srcAxes.push_back(getI64Array(axis));
  }
  int64_t rank = srcType.getRank();
  expandArrayOfArray(srcAxes, rank + 1);

  OperationName name = OperationName(getOperationName(), context);
  StringAttr meshAxisAttrName = getMeshAxisAttrName(name);
  ArrayAttr meshAxisAttr = attrs.get(meshAxisAttrName).cast<ArrayAttr>();
  SmallVector<SmallVector<int64_t>> meshAxis =
      *getArrayOfIntArray(meshAxisAttr);

  SmallVector<SmallVector<int64_t>> resAxis;
  for (size_t i = 0; i < meshAxis.size(); ++i) {
    FailureOr<SmallVector<int64_t>> maybeAxis =
        getRedIfEndWith(srcAxes[i], meshAxis[i]);
    if (failed(maybeAxis))
      return failure();
    resAxis.push_back(*maybeAxis);
  }
  for (size_t i = meshAxis.size(); i < srcAxes.size(); ++i) {
    resAxis.push_back(srcAxes[i]);
  }
  simplifyShardingOptionOrAnnotation(resAxis);
  SmallVector<ArrayAttr> resShardEncoding;
  OpBuilder b(context);
  for (const SmallVector<int64_t> &array : resAxis) {
    resShardEncoding.push_back(b.getI64ArrayAttr(array));
  }
  auto resultMeshShardingAttr =
      MeshShardingAttr::get(context, resShardEncoding);
  auto resultType = RankedTensorType::get(
      srcType.getShape(), srcType.getElementType(), resultMeshShardingAttr);
  inferredReturnTypes.push_back(resultType);

  return success();
}

//===----------------------------------------------------------------------===//
// mesh.size
//===----------------------------------------------------------------------===//

LogicalResult SizeOp::verify() {
  Operation *op = getOperation();
  FailureOr<ClusterOp> clusterOp = getMeshClusterOp(op);

  if (failed(clusterOp))
    return emitOpError("no cluster op");

  int64_t axis = getAxis().getSExtValue();
  if (axis >= (int64_t)clusterOp->getRank())
    return emitOpError("axis is expected to within the range of cluster rank");

  return success();
}

OpFoldResult SizeOp::fold(FoldAdaptor) {
  Operation *op = getOperation();
  FailureOr<ClusterOp> clusterOp = getMeshClusterOp(op);

  if (failed(clusterOp))
    return {};

  int64_t axis = getAxis().getSExtValue();
  if (axis >= (int64_t)clusterOp->getRank())
    return {};

  SmallVector<int64_t> dimSizes = getI64Array(clusterOp->getDimSizes());
  MLIRContext *ctx = op->getContext();
  return IntegerAttr::get(IndexType::get(ctx), APInt(64, dimSizes[axis]));
}

//===----------------------------------------------------------------------===//
// MeshDialect
//===----------------------------------------------------------------------===//

Operation *MeshDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
