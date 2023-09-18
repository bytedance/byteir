//===- ShardingPartition.cpp --------------------------------------- C++ --===//
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

#include "byteir/Dialect/Mesh/Transforms/ShardingPartition.h"
#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/GraphUtils.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <deque>
#include <vector>

#include "PassDetail.h"

#define DEBUG_TYPE "sharding-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

namespace {

class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

//===----------------------------------------------------------------------===//
// common util function
//===----------------------------------------------------------------------===//

FailureOr<RankedTensorType> getShardedType(RankedTensorType oldType,
                                           ArrayRef<int64_t> clusterShape) {
  Attribute encoding = oldType.getEncoding();
  MeshShardingAttr shardEncoding =
      encoding.dyn_cast_or_null<MeshShardingAttr>();
  if (!shardEncoding)
    return oldType;
  ArrayRef<ArrayAttr> axes = shardEncoding.getAxes();
  FailureOr<SmallVector<SmallVector<int64_t>>> maybeAxesArray =
      getArrayOfIntArray(axes);
  if (failed(maybeAxesArray))
    return failure();
  SmallVector<SmallVector<int64_t>> axesArray = *maybeAxesArray;
  int64_t rank = oldType.getRank();

  // get new ranked tensor type
  ArrayRef<int64_t> oldShape = oldType.getShape();
  expandArrayOfArray(axesArray, rank);
  SmallVector<int64_t> newShape;
  newShape.reserve(rank);
  for (auto it : llvm::zip(oldShape, axesArray)) {
    ArrayRef<int64_t> curAxis = std::get<1>(it);
    int64_t oldDimSize = std::get<0>(it);
    int64_t numDevices = 1;
    for (int64_t axis : curAxis) {
      if (clusterShape[axis] <= 0) {
        numDevices = ShapedType::kDynamic;
        break;
      }
      numDevices *= clusterShape[axis];
    }
    int64_t newDimSize = ShapedType::kDynamic;
    if (!ShapedType::isDynamic(oldDimSize) &&
        !ShapedType::isDynamic(numDevices) && oldDimSize % numDevices == 0) {
      newDimSize = oldDimSize / numDevices;
    }
    newShape.push_back(newDimSize);
  }
  auto newType = RankedTensorType::get(newShape, oldType.getElementType());
  return newType;
}

//===----------------------------------------------------------------------===//
// operation handle function
//===----------------------------------------------------------------------===//

LogicalResult handleArgument(OpBuilder &b, BlockArgument arg,
                             ArrayRef<int64_t> clusterShape,
                             bool allowSignatureChange) {
  auto oldType = arg.getType().cast<RankedTensorType>();
  // FIXME
  if (!allowSignatureChange)
    return failure();

  FailureOr<RankedTensorType> newType = getShardedType(oldType, clusterShape);
  if (failed(newType))
    return failure();
  arg.setType(*newType);
  return success();
}

LogicalResult handleAllGather(SimpleRewriter &rewriter,
                              mesh::AllGatherOp allGatherOp,
                              ArrayRef<int64_t> clusterShape,
                              bool allowSignatureChange) {
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<int64_t> tensorAxis = getI64Array(allGatherOp.getTensorAxis());
  SmallVector<SmallVector<int64_t>> meshAxis =
      *getArrayOfIntArray(allGatherOp.getMeshAxis());
  // FIXME
  if (tensorAxis.size() > 1)
    return failure();
  // FIXME
  if (!allowSignatureChange)
    return failure();

  auto oldResType = allGatherOp.getResult().getType().cast<RankedTensorType>();
  FailureOr<RankedTensorType> newResType =
      getShardedType(oldResType, clusterShape);
  if (failed(newResType))
    return failure();

  FailureOr<DenseIntElementsAttr> replicaGroups = mesh::getReplicaGroups(
      rewriter.getContext(), clusterShape, meshAxis[tensorAxis[0]]);
  if (failed(replicaGroups))
    return failure();

  rewriter.replaceOpWithNewOp<mhlo::AllGatherOp>(
      allGatherOp, *newResType, allGatherOp.getSrc(), tensorAxis[0],
      *replicaGroups, mhlo::ChannelHandleAttr::get(ctx, 0, 0));
  return success();
}

LogicalResult handleReduceScatter(SimpleRewriter &rewriter,
                                  mesh::ReduceScatterOp reduceScatterOp,
                                  ArrayRef<int64_t> clusterShape,
                                  bool allowSignatureChange) {
  MLIRContext *ctx = rewriter.getContext();
  int64_t tensorAxis = reduceScatterOp.getTensorAxis();
  SmallVector<int64_t> meshAxis = getI64Array(reduceScatterOp.getMeshAxis());
  // FIXME
  if (!allowSignatureChange)
    return failure();
  auto oldResType =
      reduceScatterOp.getResult().getType().cast<RankedTensorType>();
  FailureOr<RankedTensorType> newResType =
      getShardedType(oldResType, clusterShape);
  if (failed(newResType))
    return failure();

  FailureOr<DenseIntElementsAttr> replicaGroups =
      mesh::getReplicaGroups(ctx, clusterShape, meshAxis);
  if (failed(replicaGroups))
    return failure();

  auto newReduceScatterOp = rewriter.replaceOpWithNewOp<mhlo::ReduceScatterOp>(
      reduceScatterOp, *newResType, reduceScatterOp.getSrc(), tensorAxis,
      *replicaGroups, mhlo::ChannelHandleAttr::get(ctx, 0, 0));
  Location loc = reduceScatterOp.getLoc();
  // TODO: handle other reduction types
  getReduceSumBlock(rewriter, loc, newReduceScatterOp.getRegion(),
                    oldResType.getElementType());
  return success();
}

LogicalResult handleLocalSplit(SimpleRewriter &rewriter,
                               mesh::LocalSplitOp localSplitOp,
                               ArrayRef<int64_t> clusterShape,
                               bool allowSignatureChange) {
  Value src = localSplitOp.getSrc();
  mhlo::ConstantOp constOp = src.getDefiningOp<mhlo::ConstantOp>();
  // TODO: none-const case
  if (!constOp)
    return failure();

  ElementsAttr attr = constOp.getValue();
  SplatElementsAttr splatAttr = attr.dyn_cast<SplatElementsAttr>();
  // TODO: none-splat case
  if (!splatAttr)
    return failure();

  RankedTensorType oldType =
      localSplitOp.getResult().getType().cast<RankedTensorType>();
  FailureOr<RankedTensorType> newType = getShardedType(oldType, clusterShape);
  if (failed(newType))
    return failure();
  ElementsAttr newAttr = splatAttr.resizeSplat(*newType);
  rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(localSplitOp, newAttr);
  return success();
}

struct ShardingPartitionPass
    : public mlir::ShardingPartitionBase<ShardingPartitionPass> {
  ShardingPartitionPass() = default;
  ShardingPartitionPass(bool allowSignatureChange) {
    this->allowSignatureChange = allowSignatureChange;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Region &region = funcOp.getBody();
    OpBuilder builder(ctx);
    SimpleRewriter rewriter(ctx);
    if (!region.hasOneBlock())
      return signalPassFailure();
    Block &block = region.front();
    FailureOr<ClusterOp> clusterOp = getMeshClusterOp(funcOp);
    if (failed(clusterOp)) {
      funcOp.emitOpError() << "fail to get cluster op";
      return signalPassFailure();
    }
    SmallVector<int64_t> clusterShape = getI64Array(clusterOp->getDimSizes());

    // 1. modify function arguments
    for (BlockArgument arg : funcOp.getArguments()) {
      if (failed(handleArgument(builder, arg, clusterShape,
                                allowSignatureChange))) {
        funcOp.emitOpError() << "fail to process argument";
        return signalPassFailure();
      }
    }

    // 2. modify operations
    std::vector<Operation *> oldOps = getOperationsVector(block);
    for (Operation *op : oldOps) {
      if (op->hasTrait<OpTrait::IsTerminator>())
        continue;
      builder.setInsertionPoint(op);
      rewriter.setInsertionPoint(op);
      // TODO: handle other mesh ccl op types
      if (auto allGatherOp = llvm::dyn_cast<mesh::AllGatherOp>(op)) {
        if (failed(handleAllGather(rewriter, allGatherOp, clusterShape,
                                   allowSignatureChange))) {
          funcOp.emitOpError() << "fail to process all gather op";
          return signalPassFailure();
        }
      } else if (auto localSplitOp = llvm::dyn_cast<mesh::LocalSplitOp>(op)) {
        if (failed(handleLocalSplit(rewriter, localSplitOp, clusterShape,
                                    allowSignatureChange))) {
          funcOp.emitOpError() << "fail to process local split op";
          return signalPassFailure();
        }
      } else if (auto reduceScatterOp =
                     llvm::dyn_cast<mesh::ReduceScatterOp>(op)) {
        if (failed(handleReduceScatter(rewriter, reduceScatterOp, clusterShape,
                                       allowSignatureChange))) {
          funcOp.emitOpError() << "fail to process reduce scatter op";
          return signalPassFailure();
        }
      } else if (llvm::isa<ShardingInterface>(op)) {
        ValueRange results = op->getResults();
        for (Value res : results) {
          auto oldType = res.getType().dyn_cast<RankedTensorType>();
          if (!oldType) {
            op->emitOpError() << "fail to get ranked tensor type";
            return signalPassFailure();
          }
          FailureOr<RankedTensorType> newType =
              getShardedType(oldType, clusterShape);
          if (failed(newType)) {
            op->emitOpError() << "fail to get sharded type";
            return signalPassFailure();
          }
          res.setType(*newType);
        }
      } else {
        op->emitOpError() << "not supported operation type";
        return signalPassFailure();
      }
    }

    // 3. modify function signature
    if (allowSignatureChange) {
      Operation *returnOp = block.getTerminator();
      SmallVector<Type> newFuncRetTypes =
          llvm::to_vector(returnOp->getOperandTypes());
      SmallVector<Type> newFuncArgTypes =
          llvm::to_vector(block.getArgumentTypes());
      funcOp.setType(FunctionType::get(ctx, newFuncArgTypes, newFuncRetTypes));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createShardingPartitionPass(bool allowSignatureChange) {
  return std::make_unique<ShardingPartitionPass>(allowSignatureChange);
}
