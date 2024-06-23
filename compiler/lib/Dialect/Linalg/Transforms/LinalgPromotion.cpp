//===- LinalgPromotion.cpp ----------------------------------*--- C++-*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
// Some code comes from
// mlir/lib/Dialect/Linalg/Transforms/Promotion.cpp of
// LLVM project.
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code comes from
// compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp
// of IREE project
// Original licence:
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/LinalgPromotion.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;

#define DEBUG_TYPE "linalg-prefetch"

namespace {

constexpr StringRef allocMarker[3] = {getAllocSharedMemoryAMarker(),
                                      getAllocSharedMemoryBMarker(),
                                      getAllocSharedMemoryAccMarker()};
constexpr StringRef copyMarker[3] = {getCopyToSharedMemoryAMarker(),
                                     getCopyToSharedMemoryBMarker(),
                                     getCopyFromSharedMemoryAccMarker()};

namespace MatmulOperands {
constexpr static int64_t A = 0;
constexpr static int64_t B = 1;
constexpr static int64_t C = 2;
} // namespace MatmulOperands

// Insert shared memory allocation in front of the scf.forall op.
template <int OPERAND>
std::optional<Value>
allocateWorkgroupMemory(OpBuilder &builder, memref::SubViewOp subview,
                        ArrayRef<Value> sizeBounds, DataLayout &) {
  OpBuilder::InsertionGuard guard(builder);

  scf::ForallOp forallOp = subview->getParentOfType<scf::ForallOp>();
  if (!forallOp)
    return std::nullopt;

  // The subview size bounds are expected to be constant; they specify the shape
  // of the allocation.
  SmallVector<int64_t, 2> shape;
  for (Value bound : sizeBounds) {
    APInt value;
    if (!matchPattern(bound, m_ConstantInt(&value)))
      return std::nullopt;
    shape.push_back(value.getSExtValue());
  }

  builder.setInsertionPointToStart(forallOp.getBody());
  auto type = MemRefType::get(
      shape, subview.getType().getElementType(), MemRefLayoutAttrInterface{},
      gpu::AddressSpaceAttr::get(builder.getContext(),
                                 gpu::GPUDialect::getWorkgroupAddressSpace()));
  memref::AllocOp buffer =
      builder.create<memref::AllocOp>(forallOp.getLoc(), type);
  setMarker(buffer, allocMarker[OPERAND]);
  // To fix fill op. The FillOp operand `subview` should be rewrited to
  // `alloca`
  subview->replaceUsesWithIf(buffer, [&](OpOperand &opOperand) {
    return isa<linalg::FillOp>(opOperand.getOwner());
  });
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/) {
  return success();
}

// For MatmulOperands::A and MatmulOperands::B, do copy global memory.
// For MatmulOperands::C, do nothing. As we have handled Linalg FillOp before.
template <int OPERAND>
LogicalResult copyGlobalMemoryToWorkgroupMemory(OpBuilder &b, Value src,
                                                Value dst) {
  // don't copy to C, because there should be a FillOp
  if (OPERAND == MatmulOperands::C) {
    return success();
  }
  Operation *copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setLinalgTransformationMarker(copyOp,
                                getCopyRelatedToWorkgroupMemoryMarker());
  setMarker(copyOp, copyMarker[OPERAND]);
  return success();
}

LogicalResult copyWorkgroupMemoryToGlobalMemory(OpBuilder &b, Value src,
                                                Value dst) {
  OpBuilder::InsertionGuard guard(b);

  auto op = src.getDefiningOp();
  // get the only scf.for op inside the scf.forall op.
  scf::ForallOp forallOp = op->getParentOfType<scf::ForallOp>();
  auto forOps = llvm::to_vector(forallOp.getOps<scf::ForOp>());
  if (forOps.size() != 1)
    return forallOp.emitError("expected a single scf.for op");

  // copyWorkgroupMemoryToGlobalMemory after gemm compute ends.
  b.setInsertionPointAfter(forOps[0]);
  b.create<gpu::BarrierOp>(src.getLoc());
  Operation *copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setLinalgTransformationMarker(copyOp,
                                getCopyRelatedToWorkgroupMemoryMarker());
  setMarker(copyOp, copyMarker[MatmulOperands::C]);
  return success();
}

template <int OPERAND>
static linalg::LinalgPromotionOptions getPromotionOptionsForMatmulOperand() {
  linalg::LinalgPromotionOptions promotionOptions;
  promotionOptions
      .setAllocationDeallocationFns(allocateWorkgroupMemory<OPERAND>,
                                    deallocateWorkgroupMemory)
      .setCopyInOutFns(copyGlobalMemoryToWorkgroupMemory<OPERAND>,
                       copyWorkgroupMemoryToGlobalMemory)
      .setOperandsToPromote({OPERAND})
      .setUseFullTileBuffers({false, false});
  return promotionOptions;
}

template <int OPERAND>
static LogicalResult promotionImpl(OpBuilder &builder, Operation *op) {
  linalg::LinalgPromotionOptions promotionOptions =
      getPromotionOptionsForMatmulOperand<OPERAND>();

  if (failed(promoteSubviewsPrecondition(op, promotionOptions)))
    return failure();

  // PromoteSubViews will modify linalg op inplace.
  std::optional<linalg::LinalgOp> promotedOp =
      promoteSubViews(builder, cast<linalg::LinalgOp>(op), promotionOptions);
  if (!promotedOp) {
    return op->emitError("subview promotion failed");
  }
  return success();
}

// Split input/output operand from copy from shared memory into a separate
// input.
static void insertInputValueIntoGeneric(Value source,
                                        linalg::GenericOp genericOp) {
  Location loc = genericOp.getLoc();
  SmallVector<Value> inputOperands;
  SmallVector<AffineMap> operandMaps;

  // Get and add existing input operands and their corresponding indexing maps.
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    inputOperands.push_back(inputOperand->get());
    operandMaps.push_back(genericOp.getMatchingIndexingMap(inputOperand));
  }

  // Add the new input operand.
  inputOperands.push_back(source);

  // Ensure there is only one output operand.
  assert(genericOp.getNumDpsInits() == 1);
  OpOperand *outputOperand = genericOp.getDpsInitOperand(0);

  // Add indexing maps for the output operand.
  operandMaps.push_back(genericOp.getMatchingIndexingMap(outputOperand));
  operandMaps.push_back(genericOp.getMatchingIndexingMap(outputOperand));

  SmallVector<utils::IteratorType> iteratorTypes(genericOp.getNumLoops(),
                                                 utils::IteratorType::parallel);

  OpBuilder builder(genericOp);

  // Create a new GenericOp.
  auto newGenericOp = builder.create<linalg::GenericOp>(
      loc, inputOperands, outputOperand->get(), operandMaps, iteratorTypes);

  // Move the original operation's blocks to the new operation.
  newGenericOp.getRegion().getBlocks().splice(
      newGenericOp.getRegion().begin(), genericOp.getRegion().getBlocks());

  // Add a new argument to the payload.
  Block &payload = newGenericOp.getRegion().front();
  payload.addArgument(payload.getArguments().back().getType(), loc);

  // Set the Linalg transformation marker.
  setLinalgTransformationMarker(newGenericOp,
                                getCopyRelatedToWorkgroupMemoryMarker());
}

/// Propagate the shared memory copy into the consumer op if it's a fully
/// parallel linalg.generic.
static bool
propagateCopySourceIntoConsumerGeneric(linalg::CopyOp copyOp,
                                       SmallVector<Operation *> &toDelete) {
  // Look for a generic Op reading the copyOp target.
  Operation *nextOp = copyOp->getNextNode();
  while (nextOp) {
    if (isMemoryEffectFree(nextOp)) {
      nextOp = nextOp->getNextNode();
      continue;
    }
    auto consumer = dyn_cast<linalg::GenericOp>(nextOp);
    if (!consumer || consumer.getNumDpsInits() != 1 ||
        !consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0))
             .isIdentity())
      break;
    auto linalgCopyTarget = copyOp.getDpsInitOperand(0)->get();
    auto linalgCopySource = copyOp.getDpsInputOperand(0)->get();
    if (*consumer.getOutputs().begin() != linalgCopyTarget)
      break;
    insertInputValueIntoGeneric(linalgCopySource, consumer);
    toDelete.push_back(consumer);
    return true;
  }
  return false;
}

struct LinalgPromotionPass : public LinalgPromotionBase<LinalgPromotionPass> {
public:
  LinalgPromotionPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<linalg::LinalgOp> toPromote;

    if (!hasGemmTileConfig(funcOp))
      return;

    auto forallOptional = getForallOpMappedTo2DBlock(funcOp);
    if (!forallOptional)
      return;

    scf::ForallOp forallOp = *forallOptional;
    forallOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(linalgOp))
        toPromote.push_back(linalgOp);
    });
    if (toPromote.empty())
      return;

    assert(toPromote.size() == 1);
    auto linalgContractOp = toPromote[0];
    OpBuilder builder(linalgContractOp);

    // As we want to mark every generated op, so we do promote seperately.
    (void)promotionImpl<MatmulOperands::A>(builder, linalgContractOp);
    (void)promotionImpl<MatmulOperands::B>(builder, linalgContractOp);

    // TODO:
    // If we do promotion before we split K, it will be much easier.
    // The right order should be split i, j, promote C, split k, promote A\B
    // As we know linalg.matmul is in a scf.for, and the subview promotionImpl
    // inserts should be in the scf.forall op.
    auto forOp = linalgContractOp->getParentOfType<scf::ForOp>();
    builder.setInsertionPoint(forOp); // before forOp
    (void)promotionImpl<MatmulOperands::C>(builder, linalgContractOp);

    // The linalg.copy should be fused with its consumer linalg.generic.
    // So first to find linalg.copy which has marker
    // "__byteir_store_matrix_c__"
    linalg::CopyOp copyToGlobalOp;
    forallOp.walk([&](linalg::CopyOp copyOp) {
      if (hasMarker(copyOp, copyMarker[MatmulOperands::C])) {
        copyToGlobalOp = copyOp;
      }
    });
    SmallVector<Operation *> toDelete;
    if (propagateCopySourceIntoConsumerGeneric(copyToGlobalOp, toDelete)) {
      toDelete.push_back(copyToGlobalOp);
      for (Operation *op : toDelete)
        op->erase();
    }
    // as we should do synchronization after linalg.copy and before
    // linalg.matmul
    builder.setInsertionPoint(linalgContractOp);
    builder.create<gpu::BarrierOp>(linalgContractOp.getLoc());
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgPromotionPass() {
  return std::make_unique<LinalgPromotionPass>();
}