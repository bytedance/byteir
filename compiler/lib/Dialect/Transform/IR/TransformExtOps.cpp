//===- TransformExtOps.cpp ------------------------------------------------===//
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

#include "byteir/Dialect/Transform/IR/TransformExtOps.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::transform_ext;

//===---------------------------------------------------------------------===//
// Type Extensions
//===---------------------------------------------------------------------===//
namespace {
struct PDLAttributeTypeTransformParamTypeInterfaceImpl
    : public transform::TransformParamTypeInterface::ExternalModel<
          PDLAttributeTypeTransformParamTypeInterfaceImpl, pdl::AttributeType> {

  /// Accept any attribute.
  DiagnosedSilenceableFailure checkPayload(Type type, Location loc,
                                           ArrayRef<Attribute> payload) const {
    return DiagnosedSilenceableFailure::success();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Op Extensions
//
// CanonicalizeExtOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_ext::CanonicalizeExtOp::apply(
    mlir::transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  static auto applyToOne = [](Operation *op) {
    PassManager pm(op->getContext(), op->getName().getStringRef());
    pm.addPass(createCanonicalizeExtPass());
    return pm.run(op);
  };

  SmallVector<Operation *> payloadResults;
  if (auto targetHandle = getTarget()) {
    for (auto &&payload : state.getPayloadOps(targetHandle)) {
      if (failed(applyToOne(payload)))
        return DiagnosedSilenceableFailure::definiteFailure();
      payloadResults.push_back(payload);
    }
  } else {
    auto topLevel = state.getTopLevel();
    if (failed(applyToOne(topLevel)))
      return DiagnosedSilenceableFailure::definiteFailure();
    payloadResults.push_back(topLevel);
  }
  if (auto result = getResults())
    results.set(result.cast<OpResult>(), payloadResults);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// CleanupOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_ext::CleanupOp::apply(mlir::transform::TransformRewriter &rewriter,
                                mlir::transform::TransformResults &results,
                                mlir::transform::TransformState &state) {
  static auto applyToOne = [](Operation *op) {
    PassManager pm(op->getContext(), op->getName().getStringRef());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());
    pm.addPass(createCanonicalizeExtPass());
    if (op->hasTrait<OpTrait::SymbolTable>()) {
      pm.addPass(createSymbolDCEPass());
    }
    return pm.run(op);
  };

  SmallVector<Operation *> payloadResults;
  if (auto targetHandle = getTarget()) {
    for (auto &&payload : state.getPayloadOps(targetHandle)) {
      if (failed(applyToOne(payload)))
        return DiagnosedSilenceableFailure::definiteFailure();
      payloadResults.push_back(payload);
    }
  } else {
    auto topLevel = state.getTopLevel();
    if (failed(applyToOne(topLevel)))
      return DiagnosedSilenceableFailure::definiteFailure();
    payloadResults.push_back(topLevel);
  }
  if (auto result = getResults())
    results.set(result.cast<OpResult>(), payloadResults);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// DumpOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_ext::DumpOp::apply(mlir::transform::TransformRewriter &rewriter,
                             mlir::transform::TransformResults & /* result*/,
                             mlir::transform::TransformState &state) {
  llvm::errs() << getMessage() << "\n";
  if (auto targetHandle = getTarget()) {
    for (auto &&payload : state.getPayloadOps(targetHandle))
      payload->dump();
  } else {
    state.getTopLevel()->dump();
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class TransformExtDialectExtension
    : public transform::TransformDialectExtension<
          TransformExtDialectExtension> {
public:
  using Base::Base;

  void init() {
    // TODO remove unused ones
    declareDependentDialect<pdl::PDLDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<linalg_ext::LinalgExtDialect>();
    declareDependentDialect<mhlo::MhloDialect>();
#if 0
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();
#endif

    registerTransformOps<
#define GET_OP_LIST
#include "byteir/Dialect/Transform/IR/TransformExtOps.cpp.inc"
        >();

    addCustomInitializationStep([](MLIRContext *context) {
      pdl::AttributeType::attachInterface<
          PDLAttributeTypeTransformParamTypeInterfaceImpl>(*context);
    });
  }
};
} // namespace

#define GET_OP_CLASSES
#include "byteir/Dialect/Transform/IR/TransformExtOps.cpp.inc"

void mlir::transform_ext::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<TransformExtDialectExtension>();
}
