//===- OFRewriteCustomOnnxOps.td ------------------------------------------===//
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

#ifndef ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_ONNX_OPS_TD
#define ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_ONNX_OPS_TD

include "src/Dialect/ONNX/ONNX.td"  // third_party/onnx-mlir/src/Dialect/ONNX/ONNX.td

def IsQuantize: Constraint<CPred<"$0.str() == \"quantize\"">, "is a quantize op">;
def IsDequantize: Constraint<CPred<"$0.str() == \"dequantize\"">, "is a dequantize op">;

//===----------------------------------------------------------------------===//
// Quantize/Dequantize Pattern
//===----------------------------------------------------------------------===//
def RewriteQuantize : Pat<
  (ONNXCustomOp:$value $inputs, $function_name, $_, $_, $_),
  (NativeCodeCall<"createQuantizeDequantize($_builder, $_loc, $0, $1, $2)"> $inputs, $function_name, $value),
  [(IsQuantize $function_name)]
>;

def RewriteDequantize : Pat<
  (ONNXCustomOp:$value $inputs, $function_name, $_, $_, $_),
  (NativeCodeCall<"createQuantizeDequantize($_builder, $_loc, $0, $1, $2)"> $inputs, $function_name, $value),
  [(IsDequantize $function_name)]
>;

#endif // ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_ONNX_OPS_TD