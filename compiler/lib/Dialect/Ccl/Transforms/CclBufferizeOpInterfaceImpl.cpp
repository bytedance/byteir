//===- CclBufferizeOpInterfaceImpl.cpp  -----------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Ccl/Transforms/CclBufferizeOpInterfaceImpl.h"
#include "PassDetail.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "byteir/Dialect/Lccl/LcclOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

using bufferization::AliasingValueList;
using bufferization::AnalysisState;
using bufferization::BufferizableOpInterface;
using bufferization::BufferizationOptions;

struct BroadcastOpInterface
    : public BufferizableOpInterface::ExternalModel<BroadcastOpInterface,
                                                    ccl::BroadcastOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto broadcastOp = cast<ccl::BroadcastOp>(op);
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, broadcastOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();

    // Since the `getBuffer` later sets the `dynamicReplicaGroupsBuffer`, the
    // type here is `FailureOr<Value>`.It must be ensured that
    // dynamicReplicaGroupsBuffer has a value, as dynamicReplicaGroupsBuffer
    // will be used later to construct new ops.
    FailureOr<Value> dynamicReplicaGroupsBuffer = Value();
    if (broadcastOp.getDynamicReplicaGroups())
      dynamicReplicaGroupsBuffer =
          getBuffer(rewriter, broadcastOp.getDynamicReplicaGroups(), options);
    rewriter.setInsertionPoint(broadcastOp);
    rewriter.create<lccl::BroadcastOp>(
        op->getLoc(), srcBuffer.value(), dynamicReplicaGroupsBuffer.value(),
        broadcastOp.getSynchronous(), broadcastOp.getReplicaGroupsAttr(),
        broadcastOp.getUniqueIdAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 {srcBuffer.value()});
    return success();
  }
};

struct SendOpInterface
    : public BufferizableOpInterface::ExternalModel<SendOpInterface,
                                                    ccl::SendOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto sendOp = cast<ccl::SendOp>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, sendOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();
    rewriter.setInsertionPoint(sendOp);
    rewriter.create<lccl::SendOp>(
        op->getLoc(), srcBuffer.value(), sendOp.getDynamicTargetIndex(),
        sendOp.getSynchronousAttr(), sendOp.getTargetIndexAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 {srcBuffer.value()});
    return success();
  }
};

struct RecvOpInterface
    : public BufferizableOpInterface::ExternalModel<RecvOpInterface,
                                                    ccl::RecvOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto recvOp = cast<ccl::RecvOp>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, recvOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();
    rewriter.setInsertionPoint(recvOp);
    rewriter.create<lccl::RecvOp>(
        op->getLoc(), srcBuffer.value(), recvOp.getDynamicSourceIndex(),
        recvOp.getSynchronousAttr(), recvOp.getSourceIndexAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 {srcBuffer.value()});
    return success();
  }
};

struct AllReduceOpInterface
    : public BufferizableOpInterface::ExternalModel<AllReduceOpInterface,
                                                    ccl::AllReduceOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto allReduceOp = cast<ccl::AllReduceOp>(op);
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, allReduceOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();

    FailureOr<Value> dynamicReplicaGroupsBuffer = Value();
    if (allReduceOp.getDynamicReplicaGroups())
      dynamicReplicaGroupsBuffer =
          getBuffer(rewriter, allReduceOp.getDynamicReplicaGroups(), options);
    rewriter.setInsertionPoint(allReduceOp);
    auto memrefType = cast<MemRefType>(srcBuffer.value().getType());
    auto allocOp =
        rewriter.create<memref::AllocOp>(allReduceOp.getLoc(), memrefType);
    rewriter.create<lccl::AllReduceOp>(
        op->getLoc(), srcBuffer.value(), allocOp,
        dynamicReplicaGroupsBuffer.value(), allReduceOp.getSynchronousAttr(),
        allReduceOp.getReductionAttr(), allReduceOp.getReplicaGroupsAttr(),
        allReduceOp.getUniqueIdAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, {allocOp});
    return success();
  }
};

struct AllGatherOpInterface
    : public BufferizableOpInterface::ExternalModel<AllGatherOpInterface,
                                                    ccl::AllGatherOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto allGatherOp = cast<ccl::AllGatherOp>(op);
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, allGatherOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();

    FailureOr<Value> dynamicReplicaGroupsBuffer = Value();
    if (allGatherOp.getDynamicReplicaGroups())
      dynamicReplicaGroupsBuffer =
          getBuffer(rewriter, allGatherOp.getDynamicReplicaGroups(), options);
    auto srcMemrefType = cast<MemRefType>(srcBuffer.value().getType());
    SmallVector<int64_t> targetShape(srcMemrefType.getShape());
    auto axis = allGatherOp.getAxis();
    int64_t size;
    if (auto groupValue = dynamicReplicaGroupsBuffer.value()) {
      auto groupMemreyType = cast<MemRefType>(groupValue.getType());
      size = groupMemreyType.getShape()[1];
    } else {
      if (allGatherOp.getReplicaGroups().has_value() == false)
        return failure();
      size = cast<ArrayAttr>((*allGatherOp.getReplicaGroups())[0]).size();
    }
    targetShape[axis] *= size;
    auto allocType =
        MemRefType::get(targetShape, srcMemrefType.getElementType());
    rewriter.setInsertionPoint(allGatherOp);
    auto allocOp =
        rewriter.create<memref::AllocOp>(allGatherOp.getLoc(), allocType);
    rewriter.create<lccl::AllGatherOp>(
        op->getLoc(), srcBuffer.value(), allocOp,
        dynamicReplicaGroupsBuffer.value(), allGatherOp.getSynchronousAttr(),
        allGatherOp.getAxisAttr(), allGatherOp.getReplicaGroupsAttr(),
        allGatherOp.getUniqueIdAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, {allocOp});
    return success();
  }
};

struct ReduceScatterOpInterface
    : public BufferizableOpInterface::ExternalModel<ReduceScatterOpInterface,
                                                    ccl::ReduceScatterOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto reduceScatterOp = cast<ccl::ReduceScatterOp>(op);
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, reduceScatterOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();

    FailureOr<Value> dynamicReplicaGroupsBuffer = Value();
    if (reduceScatterOp.getDynamicReplicaGroups())
      dynamicReplicaGroupsBuffer = getBuffer(
          rewriter, reduceScatterOp.getDynamicReplicaGroups(), options);
    int64_t size;
    if (auto groupValue = dynamicReplicaGroupsBuffer.value()) {
      auto groupMemreyType = cast<MemRefType>(groupValue.getType());
      size = groupMemreyType.getShape()[1];
    } else {
      if (reduceScatterOp.getReplicaGroups().has_value() == false)
        return failure();
      size = cast<ArrayAttr>((*reduceScatterOp.getReplicaGroups())[0]).size();
    }
    auto srcMemrefType = cast<MemRefType>(srcBuffer.value().getType());
    SmallVector<int64_t> allocMemrefShape(srcMemrefType.getShape());
    auto axis = reduceScatterOp.getAxis();
    allocMemrefShape[axis] /= size;
    rewriter.setInsertionPoint(reduceScatterOp);
    auto allocOp = rewriter.create<memref::AllocOp>(
        reduceScatterOp.getLoc(),
        MemRefType::get(allocMemrefShape, srcMemrefType.getElementType()));
    rewriter.create<lccl::ReduceScatterOp>(
        op->getLoc(), srcBuffer.value(), allocOp,
        dynamicReplicaGroupsBuffer.value(),
        reduceScatterOp.getSynchronousAttr(),
        reduceScatterOp.getReductionAttr(), reduceScatterOp.getAxisAttr(),
        reduceScatterOp.getReplicaGroupsAttr(),
        reduceScatterOp.getUniqueIdAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, {allocOp});
    return success();
  }
};
} // namespace

void mlir::ccl::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ccl::CclDialect *dialect) {
    ccl::BroadcastOp::attachInterface<BroadcastOpInterface>(*ctx);
    ccl::SendOp::attachInterface<SendOpInterface>(*ctx);
    ccl::RecvOp::attachInterface<RecvOpInterface>(*ctx);
    ccl::AllReduceOp::attachInterface<AllReduceOpInterface>(*ctx);
    ccl::AllGatherOp::attachInterface<AllGatherOpInterface>(*ctx);
    ccl::ReduceScatterOp::attachInterface<ReduceScatterOpInterface>(*ctx);
  });
}
