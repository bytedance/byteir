//===- GPUTensorCoreVectorization.h ---------------------------*---C++ -*-===//
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
// compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUTensorCoreVectorization.cpp
// of IREE project. Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "byteir/Dialect/GPU/Transforms/GPUTensorCoreVectorization.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"

#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <optional>

#include "PassDetail.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "gpu-tensorcore-vectorization"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// static void vectorizeLinalgOps(scf::ForallOp forallOp) {
static void vectorizeLinalgOps(func::FuncOp forallOp) {
  MLIRContext *context = forallOp.getContext();
  IRRewriter rewriter(context);
  forallOp.walk([&](Operation *op) {
    if (!isa<linalg::FillOp, linalg::GenericOp, linalg::ContractionOpInterface>(
            op)) {
      return WalkResult::advance();
    }
    (void)linalg::vectorize(rewriter, op);
    return WalkResult::advance();
  });
}

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register. This is needed to get good performance on sm_80 target.
static std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract) {
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : contract.getIndexingMapsArray()[0].getResults()) {
    dims.insert(expr.cast<AffineDimExpr>().getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

/// Returns vector::ContractionOp operand's index where the result is used.
static std::optional<int>
getVectorContractOpOperandId(vector::ContractionOp contractOp,
                             OpResult result) {
  if (contractOp.getLhs() == result)
    return 0;
  if (contractOp.getRhs() == result)
    return 1;
  if (contractOp.getAcc() == result)
    return 2;
  return std::nullopt;
}

/// Returns vector::ContractionOp operand's index  where the
/// vector::TransferReadOp is consumed either consumed directly or via
/// vector::ExtractStridedSliceOp.
static std::optional<int>
getVectorContractOpOperandIdForVectorReadOp(Operation *op) {
  vector::ContractionOp contractOp;

  // Check if the vector::TransferReadOp is consumed directly by
  // vector::ContractionOp.
  if (op->use_empty())
    return std::nullopt;
  Operation *firstLevelUser = *((op->getUsers()).begin());
  if (!firstLevelUser)
    return std::nullopt;
  if (auto contractOp = dyn_cast<vector::ContractionOp>(firstLevelUser))
    return getVectorContractOpOperandId(contractOp, op->getResult(0));

  // Check if the vector::TransferReadOp is consumed indirectly by
  // vector::ContractionOp. Only check until the second level of use-def chain.
  if (firstLevelUser->use_empty())
    return std::nullopt;
  Operation *secondLevelUser = *((firstLevelUser->getUsers()).begin());
  if (!secondLevelUser)
    return std::nullopt;
  if (auto contractOp = dyn_cast<vector::ContractionOp>(secondLevelUser))
    return getVectorContractOpOperandId(contractOp,
                                        firstLevelUser->getResult(0));
  return std::nullopt;
}

/// Helper function to return native size for MMA.SYNC-based operations.
static std::optional<SmallVector<int64_t>>
getMmaNativeVectorSize(Operation *op) {
  // Shape of native Tensor Core GPU mma.sync operations.
  int64_t mmaShapeM = 16;
  int64_t mmaShapeN = 8;
  int64_t mmaShapeK;

  // Shape the mma.sync warp-level operation.
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    Type sourceType = contract.getLhsType().getElementType();

    // Set mmaShapeK based on sourceType.
    if (sourceType.isInteger(4))
      mmaShapeK = 64;
    else if (sourceType.isInteger(8))
      mmaShapeK = 32;
    else if (sourceType.isF16() || sourceType.isBF16())
      mmaShapeK = 16;
    else if (sourceType.isF32())
      mmaShapeK = 8;
    else {
      return std::nullopt;
    }

    // Initialize/set the starting dims of the ranked shape, such as batch,
    // to 1.
    SmallVector<int64_t> mmaShape(contract.getIteratorTypes().size() - 3, 1);
    mmaShape.append({mmaShapeM, mmaShapeN, mmaShapeK});
    LLVM_DEBUG({
      llvm::interleaveComma(mmaShape, DBGS() << "shape for vector.contract: ");
      llvm::dbgs() << "\n";
    });
    return mmaShape;
  }
  // Shape of warp-level vector write operation.
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    if (writeOp.getVectorType().getRank() < 2)
      return std::nullopt;
    SmallVector<int64_t> outputShape(writeOp.getVectorType().getRank() - 2, 1);
    outputShape.append({mmaShapeM, mmaShapeN});
    LLVM_DEBUG({
      llvm::interleaveComma(outputShape,
                            DBGS() << "shape for vector.xfer_write: ");
      llvm::dbgs() << "\n";
    });
    return outputShape;
  }

  // Shape of warp-level vector read (load) operation.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    auto resultVectorType =
        llvm::cast<VectorType>(readOp.getVector().getType());
    Type resultElementType = resultVectorType.getElementType();

    std::optional<int> operandId =
        getVectorContractOpOperandIdForVectorReadOp(op);
    if (!operandId) {
      LLVM_DEBUG({
        DBGS() << "Failed to get operandId for vector::xfer_read: " << *op
               << "\n";
      });
      return std::nullopt;
    }

    // Loading F16 values from Shared Memory to Registers.
    if (resultElementType.isF16() || resultElementType.isBF16()) {
      // For matrixC.
      if (*operandId == 2) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeN});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }

      // For matrixA and matrixB.
      if (*operandId == 0 || *operandId == 1) {
        // MmaSyncOp input operands: matrixA and matrixB.
        // LDSMx1, x2, x4:
        // - LDSMx1 loads a 1 tile  of 8x8.
        // - LDSMx2 loads a 2 tiles of 8x8.
        // - LDSMx4 loads a 4 tiles of 8x8. (in use)
        // IREE uses the largest tiled load, i.e., LDSMx4.

        // MmaSyncOp source operand: matrixC.
        // matrixC is also read/written in tiled block of 16x16. In the pass
        // OptimizeVectorTransfer, matrixC reads are moved above the mainloop
        // and writes are moved below the mainloop. Thus, mma.sync read/write
        // accumulator inplace.
        SmallVector<int64_t> readShape;
        readShape.append({16, 16});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
    }

    // Loading F32 values from Shared Memory to Registers.
    if (resultElementType.isF32()) {
      // Set mmaShapeK for F32 datatype mma.sync.f32.tf32.m16n8k8.
      mmaShapeK = 8;

      // For matrixC.
      if (*operandId == 2) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeN});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
      // For matrixA.
      if (*operandId == 0) {
        SmallVector<int64_t> readShape;
        readShape.append({mmaShapeM, mmaShapeK});
        LLVM_DEBUG({
          llvm::interleaveComma(readShape,
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return readShape;
      }
      // For matrixB.
      if (*operandId == 1) {
        // Do not use ldmatrix for matrixB.
        // Transfer read ops may need different shapes based on how they are
        // being used. For simplicity just match the shape used by the extract
        // strided op.
        VectorType sliceType;
        for (Operation *users : op->getUsers()) {
          auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
          if (!extract)
            return std::nullopt;
          auto vecType = llvm::cast<VectorType>(extract.getResult().getType());
          if (sliceType && sliceType != vecType)
            return std::nullopt;
          sliceType = vecType;
        }
        LLVM_DEBUG({
          llvm::interleaveComma(sliceType.getShape(),
                                DBGS() << "shape for vector.xfer_read: ");
          llvm::dbgs() << "\n";
        });
        return llvm::to_vector(sliceType.getShape());
      }
    }
  }
  LDBG("unsupported shape for " << op->getName().getStringRef());
  return std::nullopt;
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getMmaNativeVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

// This pass generate mma instead of wmma instructions.
struct GPUTensorCoreVectorizationPass
    : public GPUTensorCoreVectorizationBase<GPUTensorCoreVectorizationPass> {
  GPUTensorCoreVectorizationPass()
      : GPUTensorCoreVectorizationBase() {
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    auto context = funcOp.getContext();
    if (!hasGemmTileConfig(funcOp)) {
      return signalPassFailure();
    }
    auto forallOpOptional = getForallOpMappedTo2DBlock(funcOp);
    if (!forallOpOptional.has_value()) {
      return signalPassFailure();
    }
    scf::ForallOp forallOp = *forallOpOptional;

    {
      // Step 1(a). Vectorize (linalg to vector).
      vectorizeLinalgOps(funcOp);
      LLVM_DEBUG({
        llvm::dbgs() << "\nAfter vectorizeLinalgOps:\n";
        funcOp->dump();
      });

      RewritePatternSet contractionPatterns(context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          contractionPatterns);
      vector::populateVectorReductionToContractPatterns(contractionPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(contractionPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "\nAfter populateVectorizationPatterns:\n";
        funcOp->dump();
      });

      // Step 3. Prepare vector operations to be lowered to native tensor core
      // operations (nvgpu.mmasync, nvgpu.ldmatrix).
      RewritePatternSet vectorContractPatterns(funcOp.getContext());
      mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
          vectorContractPatterns);
      mlir::populatePrepareVectorToMMAPatterns(vectorContractPatterns,
                                               /*useMMASync=*/true);
      if (failed(applyPatternsAndFoldGreedily(
              getOperation(), std::move(vectorContractPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs()
            << "\nAfter populateCastAwayVectorLeadingOneDimPatterns and "
               "populatePrepareVectorToMMAPatterns:\n";
        funcOp->dump();
      });

      // Step2. Vector for MMA intrinsic.
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorUnrollPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "\nAfter populateVectorUnrollPattern:\n";
        funcOp->dump();
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUTensorCoreVectorizationPass() {
  return std::make_unique<GPUTensorCoreVectorizationPass>();
}
