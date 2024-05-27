//===- GPUCodeGenUtils.cpp ---------------------------------------------*--- C++
//-*-===//
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

#include <optional>

#include "byteir/Conversion/GemmCodeGen/Transforms/Transforms.h"
#include "byteir/Conversion/GemmCodeGen/Utils/GPUCodeGenUtils.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "gpu-codegen-utils"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

static constexpr int32_t kNumGPUDims = 3;

// 得到TransferWriteOp, TransferReadOp, StoreOp, LoadOp等存取Op的输入
static Value getMemrefOperand(Operation *op) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
    return transferWrite.getSource();
  }
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    return transferRead.getSource();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return storeOp.getBase();
  }
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return loadOp.getBase();
  }
  return Value();
}

struct MaskResult {
  vector::CreateMaskOp maskOp;
  vector::ExtractOp maybeExtractOp;
};

// 通用的getMask函数,这里的作用是给createAsyncGroups判断transferRead是否合法用的
static MaskResult getMask(Operation *op) {
  auto transferRead = dyn_cast<vector::TransferReadOp>(op);
  if (!transferRead || !transferRead.getMask())
    return MaskResult{};
  vector::ExtractOp maybeExtractOp =
      transferRead.getMask().getDefiningOp<vector::ExtractOp>();
  auto maskOp =
      maybeExtractOp
          ? maybeExtractOp.getVector().getDefiningOp<vector::CreateMaskOp>()
          : transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
  if (maybeExtractOp) {
    if (maybeExtractOp.getStaticPosition().size() + 1 !=
        llvm::cast<VectorType>(maskOp->getResultTypes().front()).getRank()) {
      LDBG("----mask through extract unexpected position size -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    if (maybeExtractOp.getStaticPosition().size() != 1) {
      LDBG("----only mask through 2-D -> 1-D extract supported atm -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    LDBG("----mask through extract: " << maybeExtractOp);
  }
  return MaskResult{maskOp, maybeExtractOp};
}

static Value getMaskValue(RewriterBase &rewriter, Operation *op) {
  MaskResult maskResult = getMask(op);
  if (!maskResult.maskOp)
    return Value();
  Value count = maskResult.maskOp->getOperands().back();
  vector::ExtractOp maybeExtractOp = maskResult.maybeExtractOp;
  if (maybeExtractOp) {
    assert(maybeExtractOp.getStaticPosition().size() == 1 &&
           "expected single pos");
    int64_t sliceNum = maybeExtractOp.getStaticPosition()[0];
    // int64_t sliceNum = 1;
    // TODO: to support >2-D mask + extract, and all the cmp.
    Location loc = op->getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt,
        rewriter.create<arith::ConstantIndexOp>(loc, sliceNum),
        maskResult.maskOp->getOperands().front());
    count = rewriter.create<arith::SelectOp>(loc, cmp, count, zero);
  }
  return count;
}

// 提供通用的getIndices方法
static Operation::operand_range getIndices(Operation *op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

// 得到writeOp的value
static Value getValueStored(Operation *writeOp) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(writeOp)) {
    return transferWrite.getValue();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(writeOp)) {
    return storeOp.getValueToStore();
  }
  return Value();
}

// 检查是不是连续写
static bool isContiguousStore(Operation *write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write)) {
    if (!transferWrite.getPermutationMap().isMinorIdentity() ||
        !transferWrite.isDimInBounds(0) || transferWrite.getMask()) {
      LDBG("--not a contiguous store op: " << *write);
      return false;
    }
    return true;
  }
  if (isa<vector::StoreOp>(write)) {
    return true;
  }
  LDBG("--not a store op: " << write->getName().getStringRef());
  return false;
}

// 检查是不是连续读
static bool isContiguousRead(Operation *read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read)) {
    if (!transferRead.isDimInBounds(0) ||
        !transferRead.getPermutationMap().isMinorIdentity()) {
      LDBG("--not a contiguous load op: " << *read);
      return false;
    }
    return true;
  }
  if (isa<vector::LoadOp>(read)) {
    return true;
  }
  LDBG("--not a load op: " << read->getName().getStringRef());
  return false;
}

/// Return `true` if the conversion to async copy is legal.
static bool resultsInSupportedAsyncCopy(MemRefType memrefType,
                                        Operation::operand_range indices,
                                        VectorType vecType) {
  constexpr int64_t kSupportedCpAsyncAlignmentsInBytes[3] = {4, 8, 16};
  // Condition 1: the vectory rank must be supported.
  // 必须读1维的数据
  if (vecType.hasRank() != 1) {
    LDBG("----> cp.async failed, not a 1-D vector: " << vecType);
    return false;
  }

  // Condition 2: the copy size must be supported.
  bool supportedCopySize = false;
  // 要拷贝的元素数量
  int64_t numElements = vecType.getNumElements();
  // 要拷贝的元素的类型
  Type elementType = vecType.getElementType();
  for (int64_t alignmentInBytes : kSupportedCpAsyncAlignmentsInBytes) {
    // 比较alignmentInBytes * 8 和 elementType * numElements
    if (alignmentInBytes * 8 ==
        numElements * elementType.getIntOrFloatBitWidth()) {
      supportedCopySize = true;
      break;
    }
  }
  if (!supportedCopySize) {
    LDBG("----> cp.async alignment failed, "
         << numElements << " elts * " << elementType.getIntOrFloatBitWidth()
         << "b/elem = " << numElements * elementType.getIntOrFloatBitWidth()
         << "b is not supported by cp.async");
    return false;
  }

  // TODO: Condition 3: the alignments must be supported. For cp.async the
  // NVIDIA doc (section 6.4.1) says: "The address must be naturally aligned to
  // a multiple of the access size. If an address is not properly aligned, the
  // resulting behavior is undefined.".
  return true;
}

namespace mlir {

std::optional<SmallVector<int64_t, 3>> getGemmTileSize(func::FuncOp funcOp) {
  if (funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) &&
      funcOp->hasAttr(getGemmTileConfigAttrName())) {
    auto tileConfigArray =
        funcOp->getAttrOfType<ArrayAttr>(getGemmTileConfigAttrName());
    return llvm::to_vector(
        llvm::map_range(tileConfigArray.getAsRange<IntegerAttr>(),
                        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t, 3>> getGemmBlockSize(func::FuncOp funcOp) {
  if (funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) &&
      funcOp->hasAttr(getGemmBlockSizeAttrName())) {
    auto blockSizeArray =
        funcOp->getAttrOfType<ArrayAttr>(getGemmBlockSizeAttrName());
    return llvm::to_vector(
        llvm::map_range(blockSizeArray.getAsRange<IntegerAttr>(),
                        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));
  }
  return std::nullopt;
}

std::optional<int64_t> getGemmPipelineDepth(func::FuncOp funcOp) {
  if (funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) &&
      funcOp->hasAttr(getGemmPipelineDepthAttrName())) {
    return funcOp->getAttrOfType<IntegerAttr>(getGemmPipelineDepthAttrName()).getInt();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// GPU processor IDs and sizes
//===----------------------------------------------------------------------===//
// 得到procInfo
llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getGPUThreadIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                         unsigned numDims,
                         llvm::ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] = {
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]),
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(workgroupSize[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

// distribute parallel loops to warp
// TODO(YangXinyu): I think use this function to handle matmul(2 parallel loops)
// will cause error
llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getSubgroupIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                        unsigned warpSize, unsigned numDims,
                        llvm::ArrayRef<int64_t> numSubgroups) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    mlir::Value subgroupId =
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]);
    if (i == 0) {
      mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
      subgroupId = mlir::affine::makeComposedAffineApply(
          builder, loc, d0.floorDiv(builder.getAffineConstantExpr(warpSize)),
          {subgroupId});
    }
    procInfo[numDims - 1 - i] = {
        subgroupId,
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(numSubgroups[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

// MemRefType中是不是sharedMemoryAddressSpace
bool hasSharedMemoryAddressSpace(MemRefType type) {
  return nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(type);
}

void createAsyncGroups(RewriterBase &rewriter, func::FuncOp funcOp,
                       bool useMMASync) {
  LDBG("Start asyncGroups: useMMASync=" << useMMASync);
  llvm::SmallSetVector<Operation *, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](Operation *writeOp) {
    if (!isContiguousStore(writeOp))
      return WalkResult::advance();
    LDBG("--candidate writeOp: " << *writeOp);
    Value vectorVal = getValueStored(writeOp);
    if (llvm::cast<VectorType>(vectorVal.getType()).getRank() != 1) {
      LDBG("----writeOp is not an inbounds 1-D minor identity -> Skip");
      return WalkResult::advance();
    }
    Value memrefOperand = getMemrefOperand(writeOp);
    // 向shared memory中写入
    if (!hasSharedMemoryAddressSpace(
            llvm::cast<MemRefType>(memrefOperand.getType()))) {
      LDBG("----address space is not workgroup -> Skip");
      return WalkResult::advance();
    }

    Operation *readOp = vectorVal.getDefiningOp();
    // 读入需要连续
    if (readOp == nullptr || !isContiguousRead(readOp)) {
      LDBG("----no contiguous readOp defining the writeOp -> Skip");
      return WalkResult::advance();
    }

    LDBG("--candidate readOp: " << *readOp);
    // 如果读入是一个transferReadOp
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(readOp)) {
      // 如果tranferReadOp有mask
      if (transferRead.getMask()) {
        // 如有padding, padding的元素必须是0
        auto paddingCst =
            transferRead.getPadding().getDefiningOp<arith::ConstantFloatOp>();
        if (!paddingCst || !paddingCst.value().isZero()) {
          LDBG("----read padding value is not 0.f -> Skip");
          return WalkResult::advance();
        }
        auto maskResult = getMask(transferRead);
        if (!maskResult.maskOp) {
          LDBG("----read mask is not a vector.create_mask op -> Skip: "
               << transferRead.getMask());
          return WalkResult::advance();
        }
      }
    }

    // Check whether both accesses are supported before we emit: this is
    // necessary to ensure the correctness of DeviceAsyncCopyOp.
    VectorType vecType = llvm::cast<VectorType>(vectorVal.getType());
    Value storeBase = getMemrefOperand(writeOp);
    Value loadBase = getMemrefOperand(readOp);
    if (!resultsInSupportedAsyncCopy(cast<MemRefType>(loadBase.getType()),
                                     getIndices(readOp), vecType) ||
        !resultsInSupportedAsyncCopy(cast<MemRefType>(storeBase.getType()),
                                     getIndices(writeOp), vecType))
      return WalkResult::advance();

    LDBG("--writeOp can be made async -> SUCCESS");
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<Operation *> group;
    Operation *writeOp = *copyToSharedMem.begin();
    LDBG("--START a group from: " << *writeOp);
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation *nextNode = writeOp;
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation *readOp = nextNode;
        Value memrefOperand = getMemrefOperand(readOp);
        if (!hasSharedMemoryAddressSpace(
                llvm::cast<MemRefType>(memrefOperand.getType()))) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      LDBG("----> STOP accumulating into group due to: " << *nextNode);
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    // create nvgpu::DeviceAsyncCopyOp 后， 再创建一个commit_group
    for (Operation *writeOp : group) {
      rewriter.setInsertionPoint(writeOp);
      // 找到writeOp写的Value
      Value vectorVal = getValueStored(writeOp);
      auto vectorType = llvm::cast<VectorType>(vectorVal.getType());
      int64_t numElements = vectorType.getNumElements();
      Operation *readOp = vectorVal.getDefiningOp();
      Value storeBase = getMemrefOperand(writeOp);
      Value loadBase = getMemrefOperand(readOp);
      Value mask = getMaskValue(rewriter, readOp);
      auto dstMemref = llvm::cast<MemRefType>(storeBase.getType());
      int64_t sizeInBytes =
          (dstMemref.getElementTypeBitWidth() * numElements) / 8;
      UnitAttr bypassL1 =
          useMMASync && sizeInBytes == 16 ? rewriter.getUnitAttr() : UnitAttr();
      Value token = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), storeBase,
          getIndices(writeOp), loadBase, getIndices(readOp),
          rewriter.getIndexAttr(numElements), mask,
          /*bypassL1=*/bypassL1);
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    // Value groupToken = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
    //     funcOp.getLoc(),
    //     nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), tokens);
    // TODO(YangXinyu): This is synchronized with upstream.
    // mlir/lib/Dialect/NVGPU/Transforms/CreateAsyncGroups.cpp
    // Maybe think a better way to implement.
    // Don't need to wait in our case.
    // rewriter.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
    //                                           nullptr);
    // Clean up old stores.
    for (Operation *writeOp : group)
      rewriter.eraseOp(writeOp);
  }
}

//===---------------------------------------------------------------------===//
// Replace Memref users (transitively)
//===---------------------------------------------------------------------===//

/// Replaces a `use` with the `replacement` for cases where a simple
/// substition might lead to verification errors.
static std::optional<SmallVector<Value>>
replaceNonTrivialUse(RewriterBase &rewriter, Location loc, OpOperand &use,
                     Value replacement) {
  Operation *user = use.getOwner();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(user);

  LLVM_DEBUG({
    llvm::dbgs() << "\tReplacing in user by creating new user : ";
    user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });

  if (auto castOp = dyn_cast<memref::CastOp>(user)) {
    auto replacementType = llvm::cast<MemRefType>(replacement.getType());
    auto currentResultType =
        llvm::cast<MemRefType>(castOp.getResult().getType());
    if (replacementType == currentResultType) {
      // Cast is a no op, just return the replacement.
      return SmallVector<Value>{replacement};
    }
    auto newResultType = MemRefType::get(
        currentResultType.getShape(), currentResultType.getElementType(),
        replacementType.getLayout(), replacementType.getMemorySpace());
    auto newCastOp =
        rewriter.create<memref::CastOp>(loc, newResultType, replacement);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newCastOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return SmallVector<Value>(newCastOp->result_begin(),
                              newCastOp->result_end());
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
    auto currResultType =
        llvm::cast<MemRefType>(subviewOp.getResult().getType());
    auto newSourceType = llvm::cast<MemRefType>(replacement.getType());
    SmallVector<OpFoldResult> offsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = subviewOp.getMixedStrides();
    MemRefType newResultType =
        (currResultType.getRank() != newSourceType.getRank()
             ? llvm::cast<MemRefType>(
                   memref::SubViewOp::inferRankReducedResultType(
                       currResultType.getShape(), newSourceType, offsets, sizes,
                       strides))
             : nullptr);
    auto newSubviewOp = rewriter.create<memref::SubViewOp>(
        loc, newResultType, replacement, offsets, sizes, strides);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newSubviewOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return SmallVector<Value>(newSubviewOp->result_begin(),
                              newSubviewOp->result_end());
  }
  return std::nullopt;
}

void replaceMemrefUsesAndPropagateType(RewriterBase &rewriter, Location loc,
                                       Value origValue,
                                       Value replacementValue) {
  SmallVector<std::pair<Value, Value>> worklist;
  SmallVector<Operation *> toDeleteUsers;
  worklist.push_back({origValue, replacementValue});

  while (!worklist.empty()) {
    auto [original, replacement] = worklist.pop_back_val();

    LLVM_DEBUG({
      llvm::dbgs() << "//===------------------------------------------===//\n";
      llvm::dbgs() << "Replacing : ";
      original.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });

    llvm::SmallDenseSet<OpOperand *> preservedUses;

    if (original.getType() != replacement.getType()) {
      for (OpOperand &use : original.getUses()) {
        Operation *user = use.getOwner();
        // Some uses cannot be replaced.
        if (isa<func::ReturnOp, scf::YieldOp>(user)) {
          LLVM_DEBUG({
            llvm::dbgs() << "\tUnhandled user : ";
            user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
            llvm::dbgs() << "\n";
          });
          preservedUses.insert(&use);
          continue;
        }

        // Some uses might be replace-able but require creating new versions
        // of the users to pass verification.
        std::optional<SmallVector<Value>> nonTrivialUse =
            replaceNonTrivialUse(rewriter, loc, use, replacement);
        if (nonTrivialUse) {
          // Add the results of the new users created as replacements
          // for the old users. Push this back on the to worklist.
          preservedUses.insert(&use);
          for (auto [v1, v2] :
               llvm::zip_equal(user->getResults(), nonTrivialUse.value())) {
            worklist.push_back({v1, v2});
          }
          toDeleteUsers.push_back(user);
          continue;
        }
      }
    }

    // Replace all non-preserved uses.
    rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand &use) {
      if (!preservedUses.count(&use)) {
        LLVM_DEBUG({
          llvm::dbgs() << "\t\tReplacing use in :";
          use.getOwner()->print(llvm::dbgs(),
                                OpPrintingFlags().assumeVerified());
          llvm::dbgs() << "\n";
        });
        return true;
      }
      return false;
    });
  }

  // Iterate over delete-able operations in reverse and delete if
  // there are no users.
  for (auto deleteOp : llvm::reverse(toDeleteUsers)) {
    if (deleteOp->use_empty()) {
      rewriter.eraseOp(deleteOp);
    }
  }
}

void sinkOpsInCFG(const SmallVector<Operation *> &allocs,
                  DominanceInfo &dominators) {
  for (Operation *sinkOp : allocs) {
    Block *dom = nullptr;
    for (Operation *user : sinkOp->getUsers()) {
      if (!dom) {
        dom = user->getBlock();
        // Find the block in the same region.
        while (dom->getParent() != sinkOp->getParentRegion()) {
          dom = dom->getParentOp()->getBlock();
        }
        continue;
      }
      dom = dominators.findNearestCommonDominator(dom, user->getBlock());
    }
    llvm::SmallDenseSet<Operation *> users;
    for (Operation *user : sinkOp->getUsers()) {
      while (user->getParentRegion() != sinkOp->getParentRegion()) {
        user = user->getParentOp();
      }
      users.insert(user);
    }
    Operation *firstUse = dom->getTerminator();
    for (Operation &op : dom->getOperations()) {
      if (users.count(&op)) {
        firstUse = &op;
        break;
      }
    }
    sinkOp->moveBefore(firstUse);
  }
}

/// Insert barriers and wait operations if there are allocs of a different alias
/// group before the given alloc.
static void addBarrier(func::FuncOp funcOp, Operation *alloc,
                       ArrayRef<Operation *> aliasGroup) {
  Block *entryBlock = &(*funcOp.getBlocks().begin());
  bool needBarrier = false;
  if (alloc->getBlock() != entryBlock) {
    needBarrier = true;
  } else {
    for (Operation &op : entryBlock->getOperations()) {
      if (&op == alloc)
        break;
      if (op.getNumRegions() != 0) {
        needBarrier = true;
        break;
      }
      if (isa<memref::AllocaOp>(&op) && !llvm::is_contained(aliasGroup, &op)) {
        needBarrier = true;
        break;
      }
    }
  }
  if (!needBarrier)
    return;
  OpBuilder builder(alloc);
  // TODO: make it a option if needed.
  bool hasAsyncCopies = false;
  if (hasAsyncCopies) {
    Value groupToken = builder.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        SmallVector<Value>());
    builder.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                             builder.getI32IntegerAttr(0));
  }
  builder.create<gpu::BarrierOp>(alloc->getLoc());
}

// 把 float 等type的shared mem pack 后转换成i8 alloc
void packSharedMemoryAlloc(func::FuncOp funcOp) {
  DominanceInfo dominators(funcOp);
  SmallVector<Operation *> allocs;
  funcOp.walk([&](memref::AllocaOp alloca) {
    if (hasSharedMemoryAddressSpace(alloca.getType())) {
      allocs.push_back(alloca);
    }
  });
  // First sink the alloc as low as possible in the CFG.
  sinkOpsInCFG(allocs, dominators);
  SmallVector<AliasGroup> aliasGroups;
  analyseAllocsForPacking(funcOp, allocs, aliasGroups);
  // If there is 1 or less alias group there is nothing to do.
  if (aliasGroups.size() <= 1)
    return;

  // Pack all the allocations into one i8 alloc.
  // We may need to add extra barriers to make sure we are done writting or
  // reading from the previous alias group before starting a new one.
  // TODO(Yangxinyu): analysis this
  int sz = aliasGroups.size();
  // insert barrier at last aliasGroup
  for (Operation *alloc : aliasGroups[0]) { 
    addBarrier(funcOp, alloc, aliasGroups[0]); 
  }
  // for (size_t i = 0; i < aliasGroups.size(); i++) {
  //   for (Operation *alloc : aliasGroups[i]) {
  //     addBarrier(funcOp, alloc, aliasGroups[i]);
  //   }
  // }

  OpBuilder builder(funcOp.getContext());
  packAllocs(builder, funcOp, aliasGroups);
}
} // namespace mlir