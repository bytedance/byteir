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
    CPred<" isa<ElementsAttr>($0) ">,
    CPred<" cast<ElementsAttr>($0).size() == 1 ">]>, "is a DenseElementsAttr of size one">;
def TrueBoolAttr : Constraint<CPred<"$0.getValue() == true">, "this BoolAttr should be true">;
def SameTwoValuesOrAttrs : Constraint<CPred<"$0 == $1">, "two values or attrs are actually the same">;
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">, "value has exactly one use">;

def isScalarConstantTensor :
    Constraint<CPred<"onnx_mlir::isScalarConstantTensor($_self)">,
  "Value is produced by a dense ONNXConstantOp and has size one">;
def isNoneValue :
    Constraint<CPred<"onnx_mlir::isNoneValue($_self)">,
  "Value is None Value">;
def SameTwoIntegerScalarConstantValues :
    Constraint<CPred<"onnx_mlir::getElementAttributeFromONNXValue($0).getValues<APInt>()[0].getSExtValue() == onnx_mlir::getElementAttributeFromONNXValue($1).getValues<APInt>()[0].getSExtValue()">,
  "Two integer scalar constant Value have the same integer scalar">;

//===----------------------------------------------------------------------===//
// L2Norm Pattern
//===----------------------------------------------------------------------===//
def RewriteL2NormPat1 : Pat<
  (ONNXDivOp
    $input,
    (ONNXExpandOp
      (ONNXAddOp
        (ONNXReduceL2Op $input, $axes, $keep_dims, $noop_with_empty_axes),
        (ONNXConstantOp $_, $epsilon_attr, $_, $_, $_, $_, $_, $_)
      ),
      (ONNXConstantOp $_, $_, $_, $_, $_, $_, $_, $_)  // should be the shape of $input
    )
  ),
  (NativeCodeCall<"createL2Norm($_builder, $_loc, $0, $1, $2)"> $input, $axes, $epsilon_attr),
  [(isScalarConstantTensor:$axes), (TrueBoolAttr $keep_dims), (IsOneSizeElements $epsilon_attr)]>;

def RewriteL2NormPat2 : Pat<
  (ONNXDivOp
    $input,
    (ONNXReduceL2Op $input, $axes, $keep_dims, $noop_with_empty_axes)
  ),
  (NativeCodeCall<"createL2NormWithoutEps($_builder, $_loc, $0, $1)"> $input, $axes),
  [(isScalarConstantTensor:$axes), (TrueBoolAttr $keep_dims)]>;

def RewriteL2NormPat3 : Pat<
  (ONNXDivOp
    $input,
    (ONNXExpandOp
      (ONNXClipOp
        (ONNXReduceL2Op $input, $axes, $keep_dims, $noop_with_empty_axes), $min, $max
      ),
      (ONNXConstantOp $_, $_, $_, $_, $_, $_, $_, $_)  // should be the shape of $input
    )
  ),
  (NativeCodeCall<"createL2NormWithOutsideSqrtEps($_builder, $_loc, $0, $1, $2)"> $input, $axes, $min),
  [(isScalarConstantTensor:$axes), (TrueBoolAttr $keep_dims), (isNoneValue:$max)]>;

//===----------------------------------------------------------------------===//
// Quantize/Dequantize Pattern
//===----------------------------------------------------------------------===//
def RewriteQuantize : Pat<
  (ONNXQuantizeLinearOp:$output $input, $scale, $zero_point, $axis_attr, $saturate),
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
// LogSoftmax Pattern
//===----------------------------------------------------------------------===//
def RewriteLogSoftmax : Pat<
  (ONNXLogSoftmaxOp $input, $axis_attr),
  (ONNXLogOp(NativeCodeCall<"createSoftmax($_builder, $_loc, $0, $1)"> $input, $axis_attr))>;

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
  (ONNXResizeOp:$output $input, $roi, $scale, $size, $antialias, $axes, $coordinate_transformation_mode, $cubic_coeff_a,
                $exclude_outside, $extrapolation_value, $keep_aspect_ratio_policy, $mode, $nearest_mode),
  (NativeCodeCall<"createResize($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6)"> $input, $scale, $size, $coordinate_transformation_mode, $mode, $nearest_mode, $output)>;

//===----------------------------------------------------------------------===//
// LayerNorm Pattern
//===----------------------------------------------------------------------===//
def RewriteLayerNorm : Pat<
  (ONNXAddOp
    (ONNXMulOp:$mul_result
      (ONNXDivOp
        (ONNXSubOp:$input_zero_mean_0
          $input,
          (ONNXReduceMeanOp $input, $axes_0, $_, $_)
        ),
        (ONNXSqrtOp
          (ONNXAddOp
            (ONNXReduceMeanOp
              (ONNXMulOp
                $input_zero_mean_1,
                $input_zero_mean_1
              ), $axes_1, $_, $_
            ),
            (ONNXConstantOp $_, $epsilon_attr, $_, $_, $_, $_, $_, $_)
          )
        )
      ),
      (ONNXConstantOp:$scale $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $scale, $B, $axes_0, $epsilon_attr),
  [(IsOneSizeElements $epsilon_attr),
   (isScalarConstantTensor:$axes_0), (isScalarConstantTensor:$axes_1),
   (SameTwoIntegerScalarConstantValues $axes_0, $axes_1),
   (SameTwoValuesOrAttrs $input_zero_mean_0, $input_zero_mean_1),
   (HasOneUse $mul_result),
  ]>;

def RewriteLayerNormWithNoneEps : Pat<
  (ONNXAddOp
    (ONNXMulOp:$mul_result
      (ONNXDivOp
        (ONNXSubOp:$input_zero_mean_0
          $input,
          (ONNXReduceMeanOp $input, $axes_0, $_, $_)
        ),
        (ONNXSqrtOp
          (ONNXReduceMeanOp
            (ONNXMulOp
              $input_zero_mean_1,
              $input_zero_mean_1
            ), $axes_1, $_, $_
          )
        )
      ),
      (ONNXConstantOp:$scale $_, $_, $_, $_, $_, $_, $_, $_)
    ),
    (ONNXConstantOp:$B $_, $_, $_, $_, $_, $_, $_, $_)
  ),
  (NativeCodeCall<"createLayerNormWithNoneEps($_builder, $_loc, $0, $1, $2, $3)"> $input, $scale, $B, $axes_0),
  [(isScalarConstantTensor:$axes_0), (isScalarConstantTensor:$axes_1),
   (SameTwoIntegerScalarConstantValues $axes_0, $axes_1),
   (SameTwoValuesOrAttrs $input_zero_mean_0, $input_zero_mean_1),
   (HasOneUse $mul_result),
  ]>;

def RewriteLayerNormWithoutLastAdd : Pat<
  (ONNXMulOp
    (ONNXDivOp
      (ONNXSubOp:$input_zero_mean_0
        $input,
        (ONNXReduceMeanOp $input, $axes_0, $_, $_)
      ),
      (ONNXSqrtOp
        (ONNXAddOp
          (ONNXReduceMeanOp
            (ONNXMulOp
              $input_zero_mean_1,
              $input_zero_mean_1
            ), $axes_1, $_, $_
          ),
          (ONNXConstantOp $_, $epsilon_attr, $_, $_, $_, $_, $_, $_)
        )
      )
    ),
    (ONNXConstantOp:$scale $_, $_, $_, $_, $_, $_, $_, $_)
  ),
  (NativeCodeCall<"createLayerNormWithoutLastAdd($_builder, $_loc, $0, $1, $2, $3)"> $input, $scale, $axes_0, $epsilon_attr),
  [(IsOneSizeElements $epsilon_attr),
   (isScalarConstantTensor:$axes_0), (isScalarConstantTensor:$axes_1),
   (SameTwoIntegerScalarConstantValues $axes_0, $axes_1),
   (SameTwoValuesOrAttrs $input_zero_mean_0, $input_zero_mean_1),
  ]>;

//===----------------------------------------------------------------------===//
// GeLU Pattern
//===----------------------------------------------------------------------===//
def ValueSqrtTwo : Constraint<CPred<"isSplatFPCloseTo($0, std::sqrt(2))">, "value attr is not splat value of sqrt 2">;
def ValueOne : Constraint<CPred<"isSplatFP($0, 1.0)">, "value attr is not splat value of 1.0">;
def ValueHalf : Constraint<CPred<"isSplatFPCloseTo($0, 0.5)">, "value attr is not splat value of 0.5">;
def IsTwoTimes : Constraint<Or<[
  And<[
    CPred< "isa<SplatElementsAttr>($0)" >,
    CPred< "isa<SplatElementsAttr>($1)" >,
    CPred< "isFPAttrTimesCloseTo($0.cast<SplatElementsAttr>(), $1.cast<SplatElementsAttr>(), 2.0)">]>,
  And<[
    CPred< "isa<DenseElementsAttr>($0)" >,
    CPred< "isa<DenseElementsAttr>($1)" >,
    CPred< "isFPAttrTimesCloseTo($0.cast<DenseElementsAttr>(), $1.cast<DenseElementsAttr>(), 2.0)">]>]>, "value attr $0 is not 2 times of $1">;

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

//===----------------------------------------------------------------------===//
// OneHot Pattern
//===----------------------------------------------------------------------===//

def RewriteOneHot : Pat<
  (ONNXOneHotOp:$output $indices, $depth, $values, $axis),
  (NativeCodeCall<"createOneHot($_builder, $_loc, $0, $1, $2, $3, $4)"> $indices, $depth, $values, $axis, $output)>;

#endif // ONNX_FRONTEND_SRC_CONVERSION_OF_CUSTOM_CALL_TD
