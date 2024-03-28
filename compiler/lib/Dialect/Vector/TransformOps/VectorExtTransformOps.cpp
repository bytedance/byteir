//===- VectorExtTransformOps.cpp - Implementation of Vector transform ops -===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
// Some code comes from VectorExtTransformOps.cpp in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code comes from DropUnitDims.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Vector/TransformOps/VectorExtTransformOps.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/TileUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#include <numeric>

using namespace mlir;
using namespace mlir::Vector;
using namespace mlir::scf;
using namespace mlir::tensor;
using namespace mlir::transform;

#define DEBUG_TYPE "Vector-ext-transforms"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// ConvertReductionToGPUShuffleOp
//===----------------------------------------------------------------------===//
static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           Value acc, vector::CombiningKind kind,
                           uint32_t size) {
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < size; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, input, i,
                                                 /*width=*/size,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    input = vector::makeArithReduction(builder, loc, kind, input, shuffled);
  }
  return input;
}

DiagnosedSilenceableFailure transform::ConvertReductionToGPUShuffleOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto targets = SmallVector<Operation *>(state.getPayloadOps(getTarget()));
  for (const auto &payloadOp : targets) {
    payloadOp->walk([&](vector::ReductionOp reduceOp) {
      Location loc = reduceOp.getLoc();
      auto vectorOp = reduceOp.getVector();
      // if (vectorOp.getType().getRank() > 1) {
      //   reduceOp->emitError() << "the rank of vector should equal to 1";
      //   return WalkResult::interrupt();
      // }
      rewriter.setInsertionPoint(reduceOp);

      Region *parentRegion = reduceOp->getParentRegion();
      auto argNum = parentRegion->getNumArguments();
      // if (argNum != 2) {
      //   reduceOp->emitError() << "the args of region should equal to 2";
      //   return WalkResult::interrupt();
      // }
      BlockArgument blockArg = parentRegion->getArgument(0);
      llvm::ArrayRef<int64_t> argShape =
          blockArg.getType().dyn_cast<ShapedType>().getShape();

      // auto tensorSize = rewriter.create<arith::ConstantOp>(
      //     loc, rewriter.getI32IntegerAttr(argShape.back()));
      auto laneId = rewriter.create<gpu::LaneIdOp>(loc);
      auto laneVal = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), laneId);

      // Value cond = rewriter.create<arith::CmpIOp>(
      //     loc, arith::CmpIPredicate::slt, laneVal, tensorSize);
      // scf::IfOp scfIf = rewriter.create<scf::IfOp>(loc, cond, false);

      // rewriter.setInsertionPointToStart(scfIf.getBody(0));
      // Block *parentBlock = reduceOp->getBlock();

      SmallVector<Value> extractIndex;
      for (int64_t i = 0; i < argShape.size() - 1; i++) {
        extractIndex.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      }
      extractIndex.push_back(laneId);

      auto input =
          rewriter.create<tensor::ExtractOp>(loc, blockArg, extractIndex);
      // IRRewriter rewriter(reduceOp.getContext());
      Value reduce =
          warpReduction(loc, rewriter, input, reduceOp.getAcc(),
                        reduceOp.getKind(), vectorOp.getType().getShape()[0]);
      BlockArgument blockOutput = parentRegion->getArgument(1);
      ShapedType outputShape = blockOutput.getType().dyn_cast<ShapedType>();

      // if (outputShape.getNumElements() > 1) {
      //   reduceOp->emitError() << "the shape of reduction output of should
      //   equal to 1"; return WalkResult::interrupt();
      // }
      llvm::ArrayRef<int64_t> OutputShape = outputShape.getShape();

      SmallVector<Value> insertIndex;
      for (int64_t i = 0; i < OutputShape.size(); i++) {
        insertIndex.push_back(rewriter.create<index::ConstantOp>(loc, 0));
      }
      auto insertOp = rewriter.create<tensor::InsertOp>(
          loc, reduce, blockOutput, insertIndex);

      Operation *terminator = parentRegion->back().getTerminator();
      terminator->setOperands(ValueRange{insertOp});
    });
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::ConvertReductionToGPUShuffleOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  // Indicate that the `call` handle is only read by this operation because the
  // associated operation is not erased but rather modified in-place, so the
  // reference to it remains valid.
  // onlyReadsHandle(getTarget(), effects);
  producesHandle(getODSResults(0), effects);
  // consumesHandle(getODSResults(0), effects);
  consumesHandle(getTarget(), effects);

  // Indicate that the payload is modified by this operation.
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class VectorExtTransformDialectExtension
    : public transform::TransformDialectExtension<
          VectorExtTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    // TODO remove unused ones
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "byteir/Dialect/Vector/TransformOps/VectorExtTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "byteir/Dialect/Vector/TransformOps/VectorExtTransformOps.cpp.inc"

void mlir::vector_ext::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<VectorExtTransformDialectExtension>();
}
