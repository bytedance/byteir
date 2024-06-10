//===- Bufferize.cpp ----------------------------------------------- C++ --===//
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

#include "byteir/Transforms/Bufferize.h"

#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Transforms/BufferizableOpInterfaceImpl.h"
#include "byteir/Dialect/Cat/IR/CatDialect.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "byteir/Dialect/Ccl/Transforms/CclBufferizeOpInterfaceImpl.h"
#include "byteir/Dialect/Lccl/LcclOps.h"
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "byteir/Utils/OpInterfaceUtils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/interfaces/bufferizable_op_interface_impl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
#define GEN_PASS_DEF_ONESHOTBUFFERIZE
#include "byteir/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

struct OneShotBufferizePass
    : public impl::OneShotBufferizeBase<OneShotBufferizePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<ace::AceDialect, bufferization::BufferizationDialect,
                    byre::ByreDialect, linalg::LinalgDialect,
                    linalg_ext::LinalgExtDialect, memref::MemRefDialect,
                    mhlo::MhloDialect, scf::SCFDialect, shape::ShapeDialect,
                    vector::VectorDialect, ccl::CclDialect, lccl::LcclDialect>();
    // clang-format on
    byre::registerBufferizableOpInterfaceExternalModels(registry);
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    linalg_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mhlo::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
    ccl::registerBufferizableOpInterfaceExternalModels(registry);
  }

  static bool isGPUSharedMem(MemRefType type) {
    if (auto memorySpace = llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(
            type.getMemorySpace())) {
      if (memorySpace.getValue() ==
          gpu::GPUDialect::getWorkgroupAddressSpace()) {
        return true;
      }
    }
    return false;
  }

  template <typename AllocT>
  static auto createAlloc(OpBuilder &b, Location loc, MemRefType type,
                          ValueRange dynShape, size_t bufferAlignment) {
    if (bufferAlignment != 0)
      return b
          .create<AllocT>(loc, type, dynShape,
                          b.getI64IntegerAttr(bufferAlignment))
          .getResult();
    return b.create<AllocT>(loc, type, dynShape).getResult();
  }

  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions opts;
    opts.allowReturnAllocsFromLoops = true;
    opts.bufferizeFunctionBoundaries = true;
    opts.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);
    opts.bufferAlignment = 0;
    opts.allocationFn = [](OpBuilder &b, Location loc, MemRefType type,
                           ValueRange dynShape,
                           unsigned int bufferAlignment) -> FailureOr<Value> {
      if (isGPUSharedMem(type)) {
        return createAlloc<memref::AllocaOp>(b, loc, type, dynShape,
                                             bufferAlignment);
      }
      return createAlloc<memref::AllocOp>(b, loc, type, dynShape,
                                          bufferAlignment);
    };
    // opts.deallocationFn = [](OpBuilder &b, Location loc,
    //                          Value allocatedBuffer) -> LogicalResult {
    //   if (auto bufferType =
    //           llvm::dyn_cast_or_null<MemRefType>(allocatedBuffer.getType()))
    //           {
    //     if (isGPUSharedMem(bufferType)) {
    //       return success();
    //     }
    //   }

    //   // Default buffer deallocation via DeallocOp.
    //   b.create<memref::DeallocOp>(loc, allocatedBuffer);
    //   return success();
    // };

    // deny some corner cases
    opts.opFilter.denyOperation([&](Operation *op) {
      // skip cat op
      if (isa<mlir::cat::CatOpInterface>(op))
        return true;

      return false;
    });

    ModuleOp module = getOperation();
    if (failed(bufferization::runOneShotModuleBufferize(module, opts))) {
      signalPassFailure();
    }
  }
};

// ------------------------------------------------------------------------ //
// Patch of CallOpBufferizableOpInterface
// ------------------------------------------------------------------------ //
namespace CallOpBufferizableOpInterfacePatch {
/// Return the FuncOp called by `callOp`.
static func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// All function arguments are writable. It is the responsibility of the
/// CallOp to insert buffer copies where necessary.
LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                        const BufferizationOptions &options) {
  func::CallOp callOp = cast<func::CallOp>(op);
  unsigned numResults = callOp.getNumResults();
  unsigned numOperands = callOp->getNumOperands();
  func::FuncOp funcOp = getCalledFunction(callOp);
  assert(funcOp && "expected CallOp to a FuncOp");
  FunctionType funcType = funcOp.getFunctionType();

  // Result types of the bufferized CallOp.
  SmallVector<Type> resultTypes;
  // Replacement values for the existing CallOp. These are usually the results
  // of the bufferized CallOp, unless a tensor result folds onto an operand.
  SmallVector<Value> replacementValues(numResults, Value());
  // For non-tensor results: A mapping from return val indices of the old
  // CallOp to return val indices of the bufferized CallOp.
  SmallVector<std::optional<unsigned>> retValMapping(numResults, std::nullopt);
  // Operands of the bufferized CallOp.
  SmallVector<Value> newOperands(numOperands, Value());

  // 1. Compute the result types of the new CallOp.
  for (const auto &it : llvm::enumerate(callOp.getResultTypes())) {
    unsigned returnValIdx = it.index();
    Type returnType = it.value();
    if (!returnType.isa<TensorType>()) {
      // Non-tensor values are returned.
      retValMapping[returnValIdx] = resultTypes.size();
      resultTypes.push_back(returnType);
      continue;
    }

    // Returning a memref.
    retValMapping[returnValIdx] = resultTypes.size();
    resultTypes.push_back(funcType.getResult(resultTypes.size()));
  }

  // 2. Rewrite tensor operands as memrefs based on `bufferizedFuncType`.
  for (OpOperand &opOperand : callOp->getOpOperands()) {
    unsigned idx = opOperand.getOperandNumber();
    Value tensorOperand = opOperand.get();

    // Non-tensor operands are just copied.
    if (!tensorOperand.getType().isa<TensorType>()) {
      newOperands[idx] = tensorOperand;
      continue;
    }

    // Retrieve buffers for tensor operands.
    Value buffer = newOperands[idx];
    if (!buffer) {
      FailureOr<Value> maybeBuffer =
          getBuffer(rewriter, opOperand.get(), options);
      if (failed(maybeBuffer))
        return failure();
      buffer = *maybeBuffer;
    }

    // In case the affine maps are different, we may need to use a copy if we go
    // from dynamic to static offset or stride (the canonicalization cannot know
    // at this point that it is really cast compatible).
    auto isGuaranteedCastCompatible = [](Type source, Type target) {
      MemRefType sourceMemRef = dyn_cast_or_null<MemRefType>(source);
      MemRefType targetMemRef = dyn_cast_or_null<MemRefType>(target);
      if (!sourceMemRef || !targetMemRef)
        return false;

      int64_t sourceOffset, targetOffset;
      SmallVector<int64_t, 4> sourceStrides, targetStrides;
      if (failed(
              getStridesAndOffset(sourceMemRef, sourceStrides, sourceOffset)) ||
          failed(
              getStridesAndOffset(targetMemRef, targetStrides, targetOffset)))
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

    // Caller / callee type mismatch is handled with a CastOp.
    auto memRefType = funcType.getInput(idx);
    // Since we don't yet have a clear layout story, to_memref may
    // conservatively turn tensors into more dynamic memref than necessary.
    // If the memref type of the callee fails, introduce an extra memref.cast
    // that will either canonicalize away or fail compilation until we can do
    // something better.

    // Note: If `areCastCompatible`, a cast is valid, but may fail at runtime.
    // To ensure that we only generate casts that always succeed at runtime, we
    // check a fix extra conditions in `isGuaranteedCastCompatible`.
    if (buffer.getType() != memRefType) {
      if (memref::CastOp::areCastCompatible(buffer.getType(), memRefType) &&
          isGuaranteedCastCompatible(buffer.getType(), memRefType)) {
        Value castBuffer = rewriter.create<memref::CastOp>(callOp.getLoc(),
                                                           memRefType, buffer);
        buffer = castBuffer;
      } else {
        // TODO: Create alloc_tensor ops during TensorCopyInsertion.
        AnalysisState analysisState(options);
        FailureOr<Value> tensorAlloc = allocateTensorForShapedValue(
            rewriter, op->getLoc(), opOperand.get(), options);
        if (failed(tensorAlloc))
          return failure();
        auto memrefType = MemRefType::get(
            cast<TensorType>(opOperand.get().getType()).getShape(),
            cast<TensorType>(opOperand.get().getType()).getElementType(),
            AffineMap(), cast<MemRefType>(buffer.getType()).getMemorySpace());
        buffer = rewriter.create<bufferization::ToMemrefOp>(
            op->getLoc(), memrefType, *tensorAlloc);
      }
    }
    newOperands[idx] = buffer;
  }

  // 3. Create the new CallOp.
  Operation *newCallOp = rewriter.create<func::CallOp>(
      callOp.getLoc(), funcOp.getSymName(), resultTypes, newOperands);
  newCallOp->setAttrs(callOp->getAttrs());
  // Get replacement values.
  for (unsigned i = 0; i < replacementValues.size(); ++i) {
    if (replacementValues[i])
      continue;
    replacementValues[i] = newCallOp->getResult(*retValMapping[i]);
  }

  // 4. Replace the old op with the new op.
  replaceOpWithBufferizedValues(rewriter, callOp, replacementValues);

  return success();
}
} // namespace CallOpBufferizableOpInterfacePatch

// ------------------------------------------------------------------------ //
// Patch of TensorInsertOp
// ------------------------------------------------------------------------ //
namespace TensorInsertPatch {
bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) {
  assert(isa<DestinationStyleOpInterface>(op) &&
         "expected that op implements DestinationStyleOpInterface");

  if (opOperand.getOperandNumber() == 1 &&
      cast<RankedTensorType>(opOperand.get().getType()).getRank() == 0) {
    return false;
  }

  return true;
}

} // namespace TensorInsertPatch

template <typename OpTy> static bool overwriteEntireTensor(OpTy insertSliceOp) {
  RankedTensorType destType = insertSliceOp.getDestType();
  // Dest is not read if it is entirely overwritten. E.g.:
  // tensor.insert_slice %a into %t[0][10][1] : ... into tensor<10xf32>
  bool allOffsetsZero =
      llvm::all_of(insertSliceOp.getMixedOffsets(),
                   [](OpFoldResult ofr) { return isConstantIntValue(ofr, 0); });
  bool sizesMatchDestSizes = llvm::all_of(
      llvm::enumerate(insertSliceOp.getMixedSizes()), [&](const auto &it) {
        return getConstantIntValue(it.value()) ==
               destType.getDimSize(it.index());
      });
  bool allStridesOne =
      llvm::all_of(insertSliceOp.getMixedStrides(),
                   [](OpFoldResult ofr) { return isConstantIntValue(ofr, 1); });
  return !(allOffsetsZero && sizesMatchDestSizes && allStridesOne);
}

/// Return true if the (ExtractSliceOp, InsertSliceOp) pair match (i.e.
/// equivalent operand / result and same offset/sizes/strides specification).
template <typename OpTy>
static bool areEquivalentSlices(const AnalysisState &state,
                                tensor::ExtractSliceOp extractSliceOp,
                                OpTy insertSliceOp) {
  if (!extractSliceOp || !insertSliceOp)
    return false;
  if (extractSliceOp != insertSliceOp &&
      !state.areEquivalentBufferizedValues(extractSliceOp.getSource(),
                                           insertSliceOp.getDest()))
    return false;
  if (!sameOffsetsSizesAndStrides(extractSliceOp, insertSliceOp,
                                  isEqualConstantIntOrValue))
    return false;
  return true;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
template <typename OpTy>
static bool matchesInsertDestination(const AnalysisState &state, Value value,
                                     OpTy insertSliceOp) {
  // Look for matching slices.
  auto matchesSlice = [&](Value val) {
    if (auto extractSliceOp = val.getDefiningOp<tensor::ExtractSliceOp>())
      if (areEquivalentSlices(state, extractSliceOp, insertSliceOp))
        return true;
    return false;
  };
  return static_cast<bool>(llvm::all_of(
      state.findValueInReverseUseDefChain(value, matchesSlice), matchesSlice));
}

template <typename OpTy>
static bool isNotConflictingInsertSliceLikeOp(Operation *op, OpOperand *uRead,
                                              OpOperand *uConflictingWrite,
                                              const AnalysisState &state) {
  Operation *readingOp = uRead->getOwner();
  Operation *conflictingWritingOp = uConflictingWrite->getOwner();

  // Special rules for matching ExtractSliceOp/InsertSliceOp pairs. If
  // uRead is an InsertSliceOp...
  if (auto insertSliceOp = dyn_cast<OpTy>(readingOp)) {
    // As an example, consider the following IR.
    //
    // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
    // %1 = linalg.fill %cst, %0 {inplace= [true] }
    // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
    //     {inplace= [true] }

    // TODO: Use insertSliceOp.getDestOpOperand etc. when available.
    if (uRead == &insertSliceOp->getOpOperand(1) /*dest*/ &&
        matchesInsertDestination(state, uConflictingWrite->get(),
                                 insertSliceOp))
      // Case 1: The main insight is that InsertSliceOp reads only part of
      // the destination tensor. The overwritten area is not read. If
      // uConflictingWrite writes into exactly the memory location that is
      // being read by uRead, this is not a conflict.
      //
      // In the above example:
      // uRead             = OpOperand 1 (%t) of tensor.insert_slice
      // uConflictingWrite = OpOperand 1 (%0) of linalg.fill
      //
      // The read of %t does not conflict with the write of the FillOp
      // (same aliases!) because the area that the FillOp operates on is
      // exactly the one that is *not* read via %t.
      return true;

    if (uRead == &insertSliceOp->getOpOperand(0) /*source*/ &&
        uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
        (overwriteEntireTensor(insertSliceOp) ||
         matchesInsertDestination(state, uRead->get(), insertSliceOp)))
      // Case 2: The read of the source tensor and the write to the dest
      // tensor via an InsertSliceOp is not a conflict if the read is
      // reading exactly that part of an equivalent tensor that the
      // InsertSliceOp is writing.
      //
      // In the above example:
      // uRead             = OpOperand 0 (%1) of tensor.insert_slice
      // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
      return true;
  }

  // If uConflictingWrite is an InsertSliceOp...
  if (auto insertSliceOp = dyn_cast<OpTy>(conflictingWritingOp))
    // As an example, consider the following IR.
    //
    // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
    // %1 = linalg.fill %cst, %0 {inplace= [true] }
    // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
    //     {inplace= [true] }
    // %3 = vector.transfer_read %1, %cst
    //
    // In the above example:
    // uRead             = OpOperand 0 (%1) of vector.transfer_read
    // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
    // definition        = %1
    //
    // This is not a conflict because the InsertSliceOp overwrites the
    // memory segment of %1 with the exact same data. (Effectively, there
    // is no memory write here.)
    if (uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
        state.areEquivalentBufferizedValues(uRead->get(),
                                            insertSliceOp.getSource()) &&
        matchesInsertDestination(state, insertSliceOp.getSource(),
                                 insertSliceOp))
      return true;

  return false;
}

// ------------------------------------------------------------------------ //
// Patch of TensorParallelInsertSlice
// ------------------------------------------------------------------------ //
namespace TensorParallelInsertSlicePatch {
bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) {
  auto insertSliceOp = cast<tensor::ParallelInsertSliceOp>(op);

  // The source is always read.
  if (&opOperand == &op->getOpOperand(0) /*src*/)
    return true;

  // For the destination, it depends...
  assert(&opOperand == &insertSliceOp->getOpOperand(1) && "expected dest");

  return overwriteEntireTensor(insertSliceOp);
}
bool isNotConflicting(Operation *op, OpOperand *uRead,
                      OpOperand *uConflictingWrite,
                      const AnalysisState &state) {
  return isNotConflictingInsertSliceLikeOp<tensor::ParallelInsertSliceOp>(
      op, uRead, uConflictingWrite, state);
}
} // namespace TensorParallelInsertSlicePatch
} // namespace

// TODO: removed this once upstrem fixed it
RegisterOpInterfaceOverride(
    /*Op=*/func::CallOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/bufferize,
    /*Impl=*/&CallOpBufferizableOpInterfacePatch::bufferize);
RegisterOpInterfaceOverride(
    /*Op=*/tensor::InsertOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/bufferizesToMemoryRead,
    /*Impl=*/
    &TensorInsertPatch::bufferizesToMemoryRead);
RegisterOpInterfaceOverride(
    /*Op=*/tensor::ParallelInsertSliceOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/bufferizesToMemoryRead,
    /*Impl=*/
    &TensorParallelInsertSlicePatch::bufferizesToMemoryRead);
RegisterOpInterfaceOverride(
    /*Op=*/tensor::ParallelInsertSliceOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/isNotConflicting,
    /*Impl=*/
    &TensorParallelInsertSlicePatch::isNotConflicting);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
byteir::createOneShotBufferizePass() {
  return std::make_unique<OneShotBufferizePass>();
}
