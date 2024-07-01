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
// Some code from Linalg/Transforms/BufferizableOpInterfaceImpl.cpp of LLVM
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
using namespace linalg;
using namespace linalg_ext;
using namespace mlir::bufferization;

namespace {

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult
bufferizeDestinationStyleOpInterface(RewriterBase &rewriter,
                                     DestinationStyleOpInterface op,
                                     const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasPureBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasPureTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumDpsInputs());
  for (OpOperand *opOperand : op.getDpsInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
    if (failed(buffer))
      return failure();
    newInputBuffers.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  llvm::DenseSet<Value> visited;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer))
      return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  auto newOp = cast<DestinationStyleOpInterface>(cloneWithoutRegions(
      rewriter, op, /*newResultTypes=*/TypeRange{}, newOperands));

  if (op->getNumRegions() == 1) {
    rewriter.inlineRegionBefore(op->getRegion(0), newOp->getRegion(0),
                                newOp->getRegion(0).begin());
  }

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops> struct LinalgExtOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<LinalgExtBufferizableOpInterface<Ops>>(*ctx),
     ...);
  }
};
} // namespace

namespace mlir::linalg_ext {
bool LinalgExtBufferizableOpInterfaceImpl::bufferizesToMemoryRead(
    Operation *op, OpOperand &opOperand,
    const AnalysisState & /* state*/) const {
  // Operand is read if it is used in the computation.
  if (auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op)) {
    return linalgExtOp.isOperandRead(opOperand.getOperandNumber());
  } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgOp.payloadUsesValueFromOperand(&opOperand);
  }
  return false;
}

bool LinalgExtBufferizableOpInterfaceImpl::bufferizesToMemoryWrite(
    Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
  // Operand is written to if it has an aliasing OpResult.
  auto bufferizableOp = cast<BufferizableOpInterface>(op);
  return bufferizableOp.getAliasingValues(opOperand, state).getNumAliases() !=
         0;
}

bufferization::AliasingOpOperandList
LinalgExtBufferizableOpInterfaceImpl::getAliasingOpOperands(
    Operation *op, Value value, const AnalysisState &) const {
  auto genericOp = cast<DestinationStyleOpInterface>(op);

  // The i-th OpResult may alias with the i-th "out" tensor.
  return {{genericOp.getDpsInitOperand(cast<OpResult>(value).getResultNumber()),
           BufferRelation::Equivalent}};
}

bufferization::AliasingValueList
LinalgExtBufferizableOpInterfaceImpl::getAliasingValues(
    Operation *op, OpOperand &opOperand, const AnalysisState &) const {
  auto genericOp = cast<DestinationStyleOpInterface>(op);

  // The i-th "out" tensor may alias with the i-th OpResult.
  if (genericOp.isDpsInit(&opOperand))
    return {
        {genericOp.getTiedOpResult(&opOperand), BufferRelation::Equivalent}};
  return {};
}

BufferRelation LinalgExtBufferizableOpInterfaceImpl::bufferRelation(
    Operation *op, OpResult opResult, const AnalysisState &state) const {
  return BufferRelation::Equivalent;
}

LogicalResult LinalgExtBufferizableOpInterfaceImpl::bufferize(
    Operation *op, RewriterBase &rewriter,
    const BufferizationOptions &options) const {
  return bufferizeDestinationStyleOpInterface(
      rewriter, cast<DestinationStyleOpInterface>(op), options);
}
} // namespace mlir::linalg_ext

void mlir::linalg_ext::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, linalg_ext::LinalgExtDialect *dialect) {
        // Register all Linalg structured ops. `LinalgOp` is an interface and it
        // is not possible to attach an external interface to an existing
        // interface. Therefore, attach the `BufferizableOpInterface` to all ops
        // one-by-one.
        LinalgExtOpInterfaceHelper<
#define GET_OP_LIST
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
            >::registerOpInterface(ctx);
      });
}
