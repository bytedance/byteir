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

#include "byteir/Dialect/Linalg/Transforms/LinalgPromotion.h"
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

constexpr StringRef allocMarker[3] = {"__byteir_alloca_matrix_a__",
                                      "__byteir_alloca_matrix_b__",
                                      "__byteir_alloca_accumulator__"};
constexpr StringRef copyMarker[3] = {
    "__byteir_load_matrix_a__",
    "__byteir_load_matrix_b__",
    "__byteir_store_matrix_c__",
};

static void setMarker(Operation *op, StringRef marker) {
  op->setAttr(marker, UnitAttr::get(op->getContext()));
}

//===----------------------------------------------------------------------===//
// GPU workgroup memory
//===----------------------------------------------------------------------===//

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
  memref::AllocaOp buffer =
      builder.create<memref::AllocaOp>(forallOp.getLoc(), type);
  setMarker(buffer, allocMarker[OPERAND]);
  // To fix fill op. The FillOp operand `subview` should be rewrited to `alloca`
  subview->replaceUsesWithIf(buffer, [&](OpOperand &opOperand) {
    return isa<linalg::FillOp>(opOperand.getOwner());
  });
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/) {
  return success();
}

template <int OPERAND>
LogicalResult copyGlobalMemoryToWorkgroupMemory(OpBuilder &b, Value src,
                                                Value dst) {
  if (OPERAND == 2) {
    return success();
  }
  Operation *copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, copyMarker[OPERAND]);
  return success();
}

LogicalResult copyWorkgroupMemoryToGlobalMemory(OpBuilder &b, Value src,
                                                Value dst) {
  OpBuilder::InsertionGuard guard(b);

  auto op = src.getDefiningOp();
  scf::ForallOp forallOp = op->getParentOfType<scf::ForallOp>();
  // copyWorkgroupMemoryToGlobalMemory before the GPU kernel end.
  Operation *terminator = forallOp.getBody()->getTerminator();
  b.setInsertionPoint(terminator);

  Operation *copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, copyMarker[2]);
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

  // promoteSubViews will modify op inplace.
  std::optional<linalg::LinalgOp> promotedOp =
      promoteSubViews(builder, cast<linalg::LinalgOp>(op), promotionOptions);
  if (!promotedOp) {
    return op->emitError("subview promotion failed");
  }
  return success();
}

struct LinalgPromotionPass : public LinalgPromotionBase<LinalgPromotionPass> {
public:
  LinalgPromotionPass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<linalg::LinalgOp> toPromote;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(linalgOp))
        toPromote.push_back(linalgOp);
    });

    for (auto linalgOp : toPromote) {
      OpBuilder builder(linalgOp);

      // As we want to mark every generated op, so we do promote seperately.
      (void)promotionImpl<0>(builder, linalgOp);
      (void)promotionImpl<1>(builder, linalgOp);

      // TODO:
      // If we do promotion before we split K, it will be much easier.
      // The right order should be split i, j, promote C, split k, promote A\B
      // As we know linalg.matmul is in a scf.for, and the subview promotionImpl
      // inserts should be in the scf.forall op.
      auto forOp = linalgOp->getParentOfType<scf::ForOp>();
      builder.setInsertionPoint(forOp); // before forOp
      (void)promotionImpl<2>(builder, linalgOp);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgPromotionPass() {
  return std::make_unique<LinalgPromotionPass>();
}