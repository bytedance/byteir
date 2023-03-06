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
def SameTwoValuesOrAttrs : Constraint<CPred<"$0 == $1">, "two values or attrs are actually the same">;

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

//===----------------------------------------------------------------------===//
// InstanceNorm Pattern
//===----------------------------------------------------------------------===//
def RewriteInstanceNorm : Pat<
  (ONNXInstanceNormalizationOp $input, $scale, $B, $epsilon_attr),
  (NativeCodeCall<"createLayerNormAndAffine($_builder, $_loc, $0, $1, $2, $3)"> $input, $scale, $B, $epsilon_attr)>;

//===----------------------------------------------------------------------===//
// Resize Pattern
//===----------------------------------------------------------------------===//
def RewriteResize : Pat<
  (ONNXResizeOp:$output $input, $roi, $scale, $size, $coordinate_transformation_mode, $cubic_coeff_a,
                $exclude_outside, $exttrapolation_value, $mode, $nearest_mode),
  (NativeCodeCall<"createResize($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6)"> $input, $scale, $size, $coordinate_transformation_mode, $mode, $nearest_mode, $output)>;

//===----------------------------------------------------------------------===//
// LayerNorm Pattern
//===----------------------------------------------------------------------===//
def RewriteLayerNorm : Pat<
  (ONNXAddOp
    (ONNXMulOp
      (ONNXDivOp
        (ONNXSubOp:$input_zero_mean_0
          $input,
          (ONNXReduceMeanOp $input, $axis_attr_0, $_)
        ),
        (ONNXSqrtOp
          (ONNXAddOp
            (ONNXReduceMeanOp
              (ONNXPowOp
                $input_zero_mean_1,
                (ONNXConstantOp $_, $_, $_, $_, $_, $_, $_, $_)
              ), $axis_attr_1, $_
            ),
            (ONNXConstantOp $_, $epsilon_attr, $_, $_, $_, $_, $_, $_)
          )
        )
      ),
      (ONNXConstantOp:$scale $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $scale, $B, $axis_attr_1, $epsilon_attr),
  [(IsOneSizeElements $epsilon_attr),
   (IsOneSize $axis_attr_0), (IsOneSize $axis_attr_1),
   (SameTwoValuesOrAttrs $axis_attr_0, $axis_attr_1),
   (SameTwoValuesOrAttrs $input_zero_mean_0, $input_zero_mean_1),
  ]>;

def RewriteLayerNormWithoutLastAdd : Pat<
  (ONNXMulOp
    (ONNXDivOp
      (ONNXSubOp:$input_zero_mean_0
        $input,
        (ONNXReduceMeanOp $input, $axis_attr_0, $_)
      ),
      (ONNXSqrtOp
        (ONNXAddOp
          (ONNXReduceMeanOp
            (ONNXPowOp
              $input_zero_mean_1,
              (ONNXConstantOp $_, $_, $_, $_, $_, $_, $_, $_)
            ), $axis_attr_1, $_
          ),
          (ONNXConstantOp $_, $epsilon_attr, $_, $_, $_, $_, $_, $_)
        )
      )
    ),
    (ONNXConstantOp:$scale $_, $_, $_, $_, $_, $_, $_, $_)
  ),
  (NativeCodeCall<"createLayerNormWithoutLastAdd($_builder, $_loc, $0, $1, $2, $3)"> $input, $scale, $axis_attr_1, $epsilon_attr),
  [(IsOneSizeElements $epsilon_attr),
   (IsOneSize $axis_attr_0), (IsOneSize $axis_attr_1),
   (SameTwoValuesOrAttrs $axis_attr_0, $axis_attr_1),
   (SameTwoValuesOrAttrs $input_zero_mean_0, $input_zero_mean_1),
  ]>;

//===----------------------------------------------------------------------===//
// GeLU Pattern
//===----------------------------------------------------------------------===//
def ValueSqrtTwo : Constraint<CPred<"isSplatFPCloseTo($0, std::sqrt(2))">, "value attr is not splat value of sqrt 2">;
def ValueOne : Constraint<CPred<"isSplatFP($0, 1.0)">, "value attr is not splat value of 1.0">;
def ValueHalf : Constraint<CPred<"isSplatFPCloseTo($0, 0.5)">, "value attr is not splat value of 0.5">;

def RewriteGeLU : Pat<
  (ONNXMulOp
    (ONNXMulOp
      $input,
      (ONNXAddOp
        (ONNXErfOp
          (ONNXDivOp
            $input,
            (ONNXConstantOp $_, $sqrt_two, $_, $_, $_, $_, $_, $_)
          )
        ),
        (ONNXConstantOp $_, $one, $_, $_, $_, $_, $_, $_)
      )
    ),
    (ONNXConstantOp $_, $half, $_, $_, $_, $_, $_, $_)
  ),
  (NativeCodeCall<"createGeLU($_builder, $_loc, $0)"> $input),
  [(IsOneSizeElements $sqrt_two), (IsOneSizeElements $one), (IsOneSizeElements $half),
   (ValueSqrtTwo $sqrt_two), (ValueOne $one), (ValueHalf $half)
   ]>;

def RewriteGeLUWithoutLastMul : Pat<
  (ONNXMulOp
    $input,
    (ONNXAddOp
      (ONNXErfOp
        (ONNXDivOp
          $input,
          (ONNXConstantOp $_, $sqrt_two, $_, $_, $_, $_, $_, $_)
        )
      ),
      (ONNXConstantOp $_, $one, $_, $_, $_, $_, $_, $_)
    )
  ),
  (NativeCodeCall<"createGeLUWithoutLastMul($_builder, $_loc, $0)"> $input),
  [(IsOneSizeElements $sqrt_two), (IsOneSizeElements $one),
   (ValueSqrtTwo $sqrt_two), (ValueOne $one)
   ]>;

#endif // ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_CALL_TD
