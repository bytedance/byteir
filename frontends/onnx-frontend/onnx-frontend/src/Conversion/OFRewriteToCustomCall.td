//===- OFRewriteToCustomCall.td -------------------------------------------===//
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

#ifndef ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_CALL_TD
#define ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_CALL_TD

include "src/Dialect/ONNX/ONNX.td"  // third_party/onnx-mlir/src/Dialect/ONNX/ONNX.td
                 
def IsOneSize : Constraint<CPred<"$0.size() == 1">, "is of size one">;
def IsOneSizeElements : Constraint<And<[
    CPred<" $0 ">,
    CPred<" $0.isa<ElementsAttr>() ">,
    CPred<" $0.cast<ElementsAttr>().size() == 1 ">]>, "is a DenseElementsAttr of size one">;
def TrueBoolAttr : Constraint<CPred<"$0.getValue() == true">, "this BoolAttr should be true">;

//===----------------------------------------------------------------------===//
// L2Norm Pattern
//===----------------------------------------------------------------------===//
def RewriteL2Norm : Pat<
  (ONNXDivOp
    $input,
    (ONNXExpandOp
      (ONNXAddOp
        (ONNXReduceL2Op $input, $axis_attr, $keep_dims),
        (ONNXConstantOp $_, $epsilon_attr, $_, $_, $_, $_, $_, $_)
      ),
      (ONNXConstantOp $_, $_, $_, $_, $_, $_, $_, $_)  // should be the shape of $input
    )
  ),
  (NativeCodeCall<"createL2Norm($_builder, $_loc, $0, $1, $2)"> $input, $axis_attr, $epsilon_attr),
  [(IsOneSize $axis_attr), (TrueBoolAttr $keep_dims), (IsOneSizeElements $epsilon_attr)]>;

//===----------------------------------------------------------------------===//
// Quantize/Dequantize Pattern
//===----------------------------------------------------------------------===//
def RewriteQuantize : Pat<
  (ONNXQuantizeLinearOp:$output $input, $scale, $zero_point, $axis_attr),
  (NativeCodeCall<"createQuantizeDequantize<ONNXQuantizeLinearOp>($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $scale, $zero_point, $axis_attr, $output),
  []>;

def RewriteDequantize : Pat<
  (ONNXDequantizeLinearOp:$output $input, $scale, $zero_point, $axis_attr),
  (NativeCodeCall<"createQuantizeDequantize<ONNXDequantizeLinearOp>($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $scale, $zero_point, $axis_attr, $output),
  []>;

//===----------------------------------------------------------------------===//
// Softmax Pattern
//===----------------------------------------------------------------------===//
def RewriteSoftmax : Pat<
  (ONNXSoftmaxOp $input, $axis_attr),
  (NativeCodeCall<"createSoftmax($_builder, $_loc, $0, $1)"> $input, $axis_attr)>;

#endif // ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_CALL_TD
