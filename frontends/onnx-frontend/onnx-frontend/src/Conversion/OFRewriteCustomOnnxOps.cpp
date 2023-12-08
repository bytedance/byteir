//===- OFRewriteCustomOnnxOps.cpp -----------------------------------------===//
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

#include "onnx-frontend/src/Conversion/OFRewriteCustomOnnxOps.hpp"

#include "third_party/onnx-mlir/src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace onnx_frontend;

namespace {

/// ByteIR custom call target names
#define CALL_TARGET_NAME_PREFIX "byteir."

// clang-format off
// get custom call target name
#define VALID_CUSTOM_CALL_OP(func) \
    func(dequantize, Dequantize)   \
    func(quantize, Quantize)

#define GEN_FUNCNAME(call_target_name, func_name) constexpr const char *get##func_name##Name() { return #call_target_name; }

VALID_CUSTOM_CALL_OP(GEN_FUNCNAME)

//===----------------------------------------------------------------------===//
// Quantize/Dequantize
//===----------------------------------------------------------------------===//
Value createQuantizeDequantize(PatternRewriter &rewriter, Location loc,
                               ValueRange inputs, StringAttr func_name, ValueRange outputs) {
  Value output = outputs[0];
  RankedTensorType outputType =
      output.getType().dyn_cast_or_null<RankedTensorType>();
  assert(outputType != nullptr &&
         "Quantize/Dequantize's output type must be ranked");
  Value scale = inputs[1];
  RankedTensorType scaleType =
      scale.getType().dyn_cast_or_null<RankedTensorType>();
  assert(scaleType != nullptr &&
         "Quantize/Dequantize's scale type must be ranked");
  assert(scaleType.getRank() <= 1 &&
         "Quantize/Dequantize's scale rank should be 0 or 1");
  Value zeropoint = inputs[2];
  RankedTensorType zeropointType =
      zeropoint.getType().dyn_cast_or_null<RankedTensorType>();
  assert(zeropointType != nullptr &&
         "Quantize/Dequantize's zeropoint type must be ranked");
  Type zpElementType = zeropointType.getElementType(); 
  // rewrite output type to zpElementType for quantize
  if (func_name == "quantize")
    outputType = RankedTensorType::get(
      outputType.getShape(), zpElementType);

  std::string call_target_name = std::string(CALL_TARGET_NAME_PREFIX) +
                                 func_name.str();
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{outputType},
      inputs, call_target_name, false,
      rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));
  return customCallOp.getResults()[0];
}

#include "onnx-frontend/src/Conversion/OFRewriteCustomOnnxOps.inc"

struct OFRewriteCustomOnnxOpsPass
    : public onnx_frontend::OFRewriteCustomOnnxOpsBase<
          OFRewriteCustomOnnxOpsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OFRewriteCustomOnnxOpsPass)

  OFRewriteCustomOnnxOpsPass() = default;

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    std::unordered_map<std::string,
                       llvm::SmallVector<std::unique_ptr<RewritePattern>>>
        validOpSet; // quantize, dequantize
    validOpSet[getQuantizeName()].emplace_back(
        std::make_unique<RewriteQuantize>(context));
    validOpSet[getDequantizeName()].emplace_back(
        std::make_unique<RewriteDequantize>(context));

    RewritePatternSet patterns(context);
    for (auto &op : validOpSet) {
      for (auto &pattern : op.second) {
        patterns.add(std::move(pattern));
      }
    }
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace onnx_frontend {
std::unique_ptr<mlir::Pass> createOFRewriteCustomOnnxOpsPass() {
  return std::make_unique<OFRewriteCustomOnnxOpsPass>();
}
} // namespace onnx_frontend