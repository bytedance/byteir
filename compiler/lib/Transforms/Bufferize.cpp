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
    registry.insert<ace::AceDialect, bufferization::BufferizationDialect,
                    byre::ByreDialect, linalg::LinalgDialect,
                    linalg_ext::LinalgExtDialect, memref::MemRefDialect,
                    mhlo::MhloDialect, scf::SCFDialect, shape::ShapeDialect,
                    vector::VectorDialect>();
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
  }

  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions opts;
    opts.allowReturnAllocs = true;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    opts.createDeallocs = false;
    opts.bufferAlignment = 0;

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
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
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
      MemRefType sourceMemRef = source.dyn_cast_or_null<MemRefType>();
      MemRefType targetMemRef = target.dyn_cast_or_null<MemRefType>();
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
            rewriter, op->getLoc(), opOperand.get(),
            !options.createDeallocs ||
                analysisState.isTensorYielded(opOperand.get()),
            options);
        if (failed(tensorAlloc))
          return failure();
        auto memrefType = MemRefType::get(
            opOperand.get().getType().cast<TensorType>().getShape(),
            opOperand.get().getType().cast<TensorType>().getElementType(),
            AffineMap(), buffer.getType().cast<MemRefType>().getMemorySpace());
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
} // namespace

// TODO: removed this once upstrem fixed it
RegisterOpInterfaceOverride(
    /*Op=*/func::CallOp, /*Interface=*/BufferizableOpInterface,
    /*InterfaceMethod=*/bufferize,
    /*Impl=*/&CallOpBufferizableOpInterfacePatch::bufferize);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
byteir::createOneShotBufferizePass() {
  return std::make_unique<OneShotBufferizePass>();
}
