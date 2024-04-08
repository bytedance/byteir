//===- customized_tf_to_mhlo.cc -------------------------------------------===//
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
// Some code comes from Tensorflow project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tf_mlir_ext/pipelines/customized_tf_to_mhlo.h"
#include "./passes_detail.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "tf_mlir_ext/transforms/passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "stablehlo/dialect/ChloOps.h"

#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"

using namespace mlir;
using namespace llvm;

namespace {

struct CustomizedTfToMhloPipelinePass
    : public CustomizedTfToMhloPipelineBase<CustomizedTfToMhloPipelinePass> {
  CustomizedTfToMhloPipelinePass(
      const std::vector<std::string> &customcall_ops, bool remove_control_flow,
      bool staticalize_dynamic_shape, bool stop_after_convert_to_tf_dialect,
      bool stop_after_rewrite_custom_call,
      const std::unordered_map<std::string, Attribute>
          &additional_main_func_attrs,
      bool set_assuming_to_be_true, int64_t repeat_out_batch_size) {
    this->customCallOps = customcall_ops;
    this->removeControlFlow = remove_control_flow;
    this->staticalizeDynamicShape = staticalize_dynamic_shape;
    this->stopAfterConvertToTfDialect = stop_after_convert_to_tf_dialect;
    this->stopAfterRewriteCustomCall = stop_after_rewrite_custom_call;
    this->additional_main_func_attrs = additional_main_func_attrs;
    this->setAssumingToBeTrue = set_assuming_to_be_true;
    this->setAssumingToBeTrue = set_assuming_to_be_true;
    this->repeatOutBatchSize = repeat_out_batch_size;
  }

  void runOnOperation() override {
    auto m = getOperation();
    PassManager pm(m->getContext());

    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createSCCPPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // prun useless tf node
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tf_executor::CreateTFExecutorGraphPruningPass());
    if (removeControlFlow) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::tfext::createTFSwitchMergeToIfPass());
    }
    // prun useless tf node
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tf_executor::CreateTFExecutorGraphPruningPass());

    pm.addPass(mlir::createInlinerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateDropWhileShapeInvariantPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // The SCCP pass performs constant propagation across the IR, which, for
    // example, propagates constant arguments into callee functions.
    // TOOD(hinsu): Investigate if we really need SCCP pass before shape
    // inference and can do with just one pass after the shape inference.
    pm.addPass(mlir::createSCCPPass());
    // Guarantee all functions have one use, which enables shape inference.
    pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
    // Run shape inference pass before tensorlist decomposition to get buffer
    // shape of uninitialized TensorLists.
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());

    // Run SCCP pass again as the availability of shapes may open up new
    // opportunities for constant propagation. Note that the shape inference
    // pass doesn't materialize new constants even if those are computed
    // internally for the purpose of shape inference. These constants might be
    // required by the legalization passes.
    pm.addPass(mlir::createSCCPPass());
    pm.addPass(mlir::TF::CreateTensorListOpsDecompositionPass());
    pm.addPass(mlir::TF::CreateStackOpsDecompositionPass());
    pm.addPass(mlir::TF::CreateTensorArrayOpsDecompositionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TFDevice::CreateDecomposeResourceOpsPass());
    pm.addPass(mlir::TF::CreatePromoteResourcesToArgsPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    //// TODO(b/171426148): We cannot completely remove region to functional
    //// control flow conversion from this pipeline yet as it causes some unit
    //// tests to fail.
    pm.addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());
    //  LegalizeTFControlFlow encapsulates arguments for control flow operations
    //  with a tuple argument which break the assumption of resource lifting
    //  inside PromoteResourcesToArgs.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::CreateExecutorDialectToFunctionalConversionPass());
    if (this->stopAfterConvertToTfDialect) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::tfext::createRewriteFuncAttrToByteIRPass(
              additional_main_func_attrs));

      if (mlir::failed(runPipeline(pm, m))) {
        signalPassFailure();
      }
      return;
    }

    if (staticalizeDynamicShape) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::tfext::createProcessDynamicStitchAsStaticPass());
    }
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tfext::createReshapeMovedownStringPass());

    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tfext::createConstantFoldingPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateLowerQuantizedPass());

    // fuse dilated conv
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateIdentifyDilatedConvPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::tfext::createFuseTFOpsPass());

    pm.addPass(mlir::tfext::createRewriteToCustomCallOpsPass(
        customCallOps,
        /*keepBody*/ false, repeatOutBatchSize));

    if (this->stopAfterRewriteCustomCall) {
      pm.addPass(mlir::TF::CreateTFShapeInferencePass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::tfext::createRewriteFuncAttrToByteIRPass(
              additional_main_func_attrs));

      if (mlir::failed(runPipeline(pm, m))) {
        signalPassFailure();
      }
      return;
    }

    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tfext::createMhloLegalizeTfExtPass());
    pm.addPass(mlir::mhlo::createLegalizeTFPass(
        /*legalize_chlo=*/true,
        /*tf2xla_fallback_device_type=*/std::nullopt, false));
    pm.addPass(mlir::mhlo::CreateLegalizeTFCommunicationPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

    // Run shape inference pass to propagate shapes through tensor_cast
    // operations from static to dynamic shapes. This could be generated if the
    // shape inference was originally missing in a TF op but the corresponding
    // HLO op had static shape after lowering.
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    // Run LegalizeTFPass again because the previous legalization passes can
    // expose more graph pruning and canonicalization opportunities that are
    // necessary for the second LegalizeTFPass(allow_partial_conversion=false)
    // invocation.
    pm.addPass(mlir::tfext::createRewriteToCustomCallOpsPass(
        customCallOps,
        /*keepBody*/ false, repeatOutBatchSize));
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tfext::createMhloLegalizeTfExtPass());
    pm.addPass(mlir::mhlo::createLegalizeTFPass(
        /*legalize_chlo=*/true,
        /*tf2xla_fallback_device_type=*/std::nullopt, false));

    // if (CanInlineFunctionsPostLegalization(device_type))
    //   pm.addPass(mlir::createInlinerPass());
    // In order to export to XLA, we must sink constants to control flow
    // regions, since XLA uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createSinkConstantsToControlFlowPass());

    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createCSEPass());

    // Sparse Conditional Constant Propagation
    pm.addPass(mlir::createSCCPPass());
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(mlir::tfext::createRewriteToCustomCallOpsPass(
        customCallOps,
        /*keepBody*/ false, repeatOutBatchSize));
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tfext::createMhloLegalizeTfExtPass());
    pm.addPass(mlir::mhlo::createLegalizeTFPass(
        /*legalize_chlo=*/true,
        /*tf2xla_fallback_device_type=*/std::nullopt, false));

    pm.addPass(mlir::createInlinerPass());

    // Fallback pass to lower all ops that are not legalized to mhlo
    // to mhlo::custom_call or ace::custom_call, this pass must be after all
    // LegalizeTFPass
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tfext::createTfFallbackToCustomCallPass());

    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::tfext::createRewriteFuncAttrToByteIRPass(
            additional_main_func_attrs));

    if (setAssumingToBeTrue) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::createRemoveShapeConstraintsPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::tfext::createRemoveCstrReshapablePass());
    }
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());

    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }

private:
  std::unordered_map<std::string, Attribute> additional_main_func_attrs;
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tfext::createCustomizedTfToMhloPipelinePass(
    const std::vector<std::string> &customcall_ops /*= {}*/,
    bool remove_control_flow /*= false*/,
    bool staticalize_dynamic_shape /*= false*/,
    bool stop_after_convert_to_tf_dialect /*= false*/,
    bool stop_after_rewrite_custom_call /*= false*/,
    const std::unordered_map<std::string, Attribute>
        &additional_main_func_attrs /*= {}*/,
    bool set_assuming_to_be_true /*= true*/,
    int64_t repeat_out_batch_size /*= -1*/) {
  return std::make_unique<CustomizedTfToMhloPipelinePass>(
      customcall_ops, remove_control_flow, staticalize_dynamic_shape,
      stop_after_convert_to_tf_dialect, stop_after_rewrite_custom_call,
      additional_main_func_attrs, set_assuming_to_be_true,
      repeat_out_batch_size);
}
