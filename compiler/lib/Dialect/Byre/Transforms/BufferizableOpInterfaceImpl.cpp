//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
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

#include "byteir/Dialect/Byre/Transforms/BufferizableOpInterfaceImpl.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Utils/Utils.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::bufferization;

namespace {
FailureOr<Value> getBufferInValidLayout(RewriterBase &rewriter, Location loc,
                                        OpOperand &opOperand,
                                        const BufferizationOptions &options) {
  // In case the affine maps are different, we may need to use a copy if we go
  // from dynamic to static offset or stride (the canonicalization cannot know
  // at this point that it is really cast compatible).
  static auto isGuaranteedCastCompatible = [](Type source, Type target) {
    MemRefType sourceMemRef = source.dyn_cast_or_null<MemRefType>();
    MemRefType targetMemRef = target.dyn_cast_or_null<MemRefType>();
    if (!sourceMemRef || !targetMemRef)
      return false;

    int64_t sourceOffset, targetOffset;
    SmallVector<int64_t, 4> sourceStrides, targetStrides;
    if (failed(
            getStridesAndOffset(sourceMemRef, sourceStrides, sourceOffset)) ||
        failed(getStridesAndOffset(targetMemRef, targetStrides, targetOffset)))
      return false;
    auto dynamicToStatic = [](int64_t a, int64_t b) {
      return ShapedType::isDynamic(a) && !ShapedType::isDynamic(b);
    };
    if (dynamicToStatic(sourceOffset, targetOffset))
      return false;
    for (auto it : zip(sourceStrides, targetStrides))
      if (dynamicToStatic(std::get<0>(it), std::get<1>(it)))
        return false;
    return true;
  };

  auto bufferOrNot = getBuffer(rewriter, opOperand.get(), options);
  if (failed(bufferOrNot))
    return failure();
  auto buffer = *bufferOrNot;

  auto memRefType = MemRefType::get(
      opOperand.get().getType().cast<TensorType>().getShape(),
      opOperand.get().getType().cast<TensorType>().getElementType(),
      AffineMap(), buffer.getType().cast<MemRefType>().getMemorySpace());

  // TODO: check whether buffer is in valid layout map, e.g. identity layout map
  if (buffer.getType() != memRefType) {
    if (memref::CastOp::areCastCompatible(buffer.getType(), memRefType) &&
        isGuaranteedCastCompatible(buffer.getType(), memRefType)) {
      Value castBuffer =
          rewriter.create<memref::CastOp>(loc, memRefType, buffer);
      buffer = castBuffer;
    } else {
      // TODO: Create alloc_tensor ops during TensorCopyInsertion.
      AnalysisState analysisState(options);
      FailureOr<Value> tensorAlloc =
          allocateTensorForShapedValue(rewriter, loc, opOperand.get(), options);
      if (failed(tensorAlloc))
        return failure();
      buffer = rewriter.create<bufferization::ToMemrefOp>(loc, memRefType,
                                                          *tensorAlloc);
    }
  }
  return buffer;
}

struct ByreComputeOnTensorOpBufferization
    : public BufferizableOpInterface::ExternalModel<
          ByreComputeOnTensorOpBufferization, byre::ComputeOnTensorOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto computeOnTensorOp = cast<byre::ComputeOnTensorOp>(op);
    size_t idx = opOperand.getOperandNumber();
    byre::MemoryEffect effect =
        llvm::to_vector(computeOnTensorOp.getMemoryEffects()
                            ->getAsValueRange<byre::MemoryEffectAttr>())[idx];
    return bitEnumContainsAll(effect, byre::MemoryEffect::Read);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto computeOnTensorOp = cast<byre::ComputeOnTensorOp>(op);
    size_t idx = opOperand.getOperandNumber();
    byre::MemoryEffect effect =
        llvm::to_vector(computeOnTensorOp.getMemoryEffects()
                            ->getAsValueRange<byre::MemoryEffectAttr>())[idx];
    return bitEnumContainsAll(effect, byre::MemoryEffect::Write);
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState & /*state*/) const {
    auto genericOp = cast<DestinationStyleOpInterface>(op);
    // The i-th "out" tensor may alias with the i-th OpResult.
    if (genericOp.isDpsInit(&opOperand))
      return {
          {genericOp.getTiedOpResult(&opOperand), BufferRelation::Equivalent}};
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto DpsOp = cast<DestinationStyleOpInterface>(op);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);

    if (DpsOp.hasBufferSemantics())
      return success();

    if (!DpsOp.hasTensorSemantics())
      return DpsOp->emitError() << "op does not have tensor semantics";

    SmallVector<Value> newInputBuffers;
    for (OpOperand *opOperand : DpsOp.getDpsInputOperands()) {
      if (DpsOp.isScalar(opOperand)) {
        newInputBuffers.push_back(opOperand->get());
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
      if (failed(buffer))
        return failure();
      newInputBuffers.push_back(*buffer);
    }

    SmallVector<Value> newOutputBuffers;
    for (OpResult opResult : DpsOp->getOpResults()) {
      OpOperand *opOperand =
          DpsOp.getDpsInitOperand(opResult.getResultNumber());
      FailureOr<Value> resultBuffer =
          getBuffer(rewriter, opOperand->get(), options);
      if (failed(resultBuffer))
        return failure();
      newOutputBuffers.push_back(*resultBuffer);
    }

    rewriter.setInsertionPoint(op);

    // Convert ComputeOnTensorOp to ComputeOp
    auto newOp = rewriter.create<byre::ComputeOp>(
        op->getLoc(), cast<byre::ComputeOnTensorOp>(op).getCallee(),
        newInputBuffers, newOutputBuffers);

    for (auto &&namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName();
      if ((!name.startswith("bufferization.") &&
           name != "operandSegmentSizes") &&
          !newOp->hasAttr(name)) {
        newOp->setAttr(name, namedAttr.getValue());
      }
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 newOutputBuffers);

    return success();
  }
};

struct ByreComputeOpBufferization
    : public BufferizableOpInterface::ExternalModel<ByreComputeOpBufferization,
                                                    byre::ComputeOp> {
  bool bufferizesToAllocation(Operation * /*op*/, Value /*value*/) const {
    return true;
  }

  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation * /*op*/,
                                      OpOperand & /*opOperand*/,
                                      const AnalysisState & /*state*/) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    SmallVector<Value> bufferOperands, bufferResults;

    for (auto &&opOperand : op->getOpOperands()) {
      auto buffer =
          getBufferInValidLayout(rewriter, op->getLoc(), opOperand, options);
      if (failed(buffer))
        return failure();

      bufferOperands.push_back(*buffer);
    }

    for (auto &&opResult : op->getOpResults()) {
      auto tensorType = opResult.getType().dyn_cast_or_null<RankedTensorType>();
      if (!tensorType)
        return failure();

      auto tensorAlloc = allocateTensorForShapedValue(rewriter, op->getLoc(),
                                                      opResult, options);
      if (failed(tensorAlloc))
        return failure();

      auto memrefType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      Value buffer = rewriter.create<bufferization::ToMemrefOp>(
          op->getLoc(), memrefType, *tensorAlloc);
      bufferResults.push_back(buffer);
    }

    auto newOp = rewriter.create<byre::ComputeOp>(
        op->getLoc(), cast<byre::ComputeOp>(op).getCallee(), bufferOperands,
        bufferResults);

    for (auto &&namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName();
      if (!name.startswith("bufferization.") && !newOp->hasAttr(name)) {
        newOp->setAttr(name, namedAttr.getValue());
      }
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op, bufferResults);
    return success();
  }
};

struct ByreCustomOpBufferization
    : public BufferizableOpInterface::ExternalModel<ByreCustomOpBufferization,
                                                    byre::CustomOp> {
  bool bufferizesToAllocation(Operation * /*op*/, Value /*value*/) const {
    return true;
  }

  bool bufferizesToMemoryRead(Operation * /*op*/, OpOperand & /*opOperand*/,
                              const AnalysisState & /*state*/) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation * /*op*/, OpOperand & /*opOperand*/,
                               const AnalysisState & /*state*/) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation * /*op*/,
                                      OpOperand & /*opOperand*/,
                                      const AnalysisState & /*state*/) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    SmallVector<Value> bufferOperands, bufferResults;

    for (auto &&opOperand : op->getOpOperands()) {
      auto buffer =
          getBufferInValidLayout(rewriter, op->getLoc(), opOperand, options);
      if (failed(buffer))
        return failure();

      bufferOperands.push_back(*buffer);
    }

    for (auto &&opResult : op->getOpResults()) {
      auto tensorType = opResult.getType().dyn_cast_or_null<RankedTensorType>();
      if (!tensorType)
        return failure();

      auto tensorAlloc = allocateTensorForShapedValue(rewriter, op->getLoc(),
                                                      opResult, options);
      if (failed(tensorAlloc))
        return failure();

      auto memrefType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      Value buffer = rewriter.create<bufferization::ToMemrefOp>(
          op->getLoc(), memrefType, *tensorAlloc);
      bufferResults.push_back(buffer);
    }

    auto newOp = rewriter.create<byre::CustomOp>(
        op->getLoc(), cast<byre::CustomOp>(op).getLibPath(),
        cast<byre::CustomOp>(op).getApiName(),
        cast<byre::CustomOp>(op).getVersion(), bufferOperands, bufferResults,
        cast<byre::CustomOp>(op).getExtraArgs());

    for (auto &&namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName();
      if (!name.startswith("bufferization.") && !newOp->hasAttr(name)) {
        newOp->setAttr(name, namedAttr.getValue());
      }
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op, bufferResults);
    return success();
  }
};
} // namespace

void mlir::byre::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, byre::ByreDialect *) {
    byre::ComputeOp::attachInterface<ByreComputeOpBufferization>(*ctx);
    byre::ComputeOnTensorOp::attachInterface<
        ByreComputeOnTensorOpBufferization>(*ctx);
    byre::CustomOp::attachInterface<ByreCustomOpBufferization>(*ctx);
  });
}
