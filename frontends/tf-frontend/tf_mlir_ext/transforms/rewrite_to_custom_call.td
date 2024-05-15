//===- rewrite_to_custom_call.td -------------------------*--- tablegen -*-===//
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

#ifndef REWRITE_TO_CUSTOM_CALL_PATTERN
#define REWRITE_TO_CUSTOM_CALL_PATTERN

include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mhlo/IR/hlo_ops.td"

def OneSize : Constraint<CPred<"$0.size() == 1">, "one size">;
def TrueBoolAttr : Constraint<CPred<"$0.getValue() == true">, "this BoolAttr should be true">;
def SameTwoValuesOrAttrs : Constraint<CPred<"$0 == $1">, "two values or attrs are the same">;
def OneRank : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 1">, "ElementsAttr type' rank should be 1">;
def ThreeDimBatch : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 3 && $0.getType().cast<ShapedType>().getShape()[1] == 1">, "batch size should be 1">;

//===----------------------------------------------------------------------===//
// LayerNorm Pattern
//===----------------------------------------------------------------------===//

def RewriteLayerNorm : Pat<
  (TF_AddV2Op
    (TF_MulOp
      $input,
      (TF_MulOp:$mul_after_rsqrt
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_ConstOp $epsilon_attr),
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                $input,
                (TF_MeanOp:$mean_input
                  $input,
                  (TF_ConstOp:$axis $axis_attr),
                  $keep_dims_0
                )
              ),
              $axis_1,
              $keep_dims_1
            )
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_SubOp
      (TF_ConstOp:$beta $beta_attr),
      (TF_MulOp $mul_after_rsqrt_1, $mean_input_1)
    )
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mul_after_rsqrt, $mul_after_rsqrt_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormWithoutBeta : Pat<
  (TF_SubOp
    (TF_MulOp
      $input,
      (TF_MulOp:$mul_after_rsqrt
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                $input,
                (TF_MeanOp:$mean_input
                  $input,
                  (TF_ConstOp:$axis $axis_attr),
                  $keep_dims_0
                )
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_MulOp $mul_after_rsqrt_1, $mean_input_1)
  ),
  (NativeCodeCall<"createLayerNormWithoutBeta($_builder, $_loc, $0, $1, $2, $3)"> $input, $gama, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mul_after_rsqrt, $mul_after_rsqrt_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr)]>;

def RewriteLayerNormThreeDim : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_MulOp:$mul_after_rsqrt
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                (TF_MeanOp:$mean_input
                  $input,
                  (TF_ConstOp:$axis $axis_attr),
                  $keep_dims_0
                ),
                $input
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      ),
      $input
    ),
    (TF_SubOp
      (TF_ConstOp:$beta $beta_attr),
      (TF_MulOp $mul_after_rsqrt_1, $mean_input_1)
    )
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mul_after_rsqrt, $mul_after_rsqrt_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (ThreeDimBatch $gama_attr), (ThreeDimBatch $beta_attr)]>;

def RewriteLayerNormWithBody : Pat<
  (TF_AddV2Op:$add_outer
    (TF_MulOp:$mul_input
      $input,
      (TF_MulOp:$mul_after_rsqrt
        (TF_RsqrtOp:$rsqrt
          (TF_AddV2Op:$add_mean
            (TF_ConstOp:$const_epsilon $epsilon_attr),
            (TF_MeanOp:$mean_sqr
              (TF_SquaredDifferenceOp:$sqdiff
                $input,
                (TF_MeanOp:$mean_input
                  $input,
                  (TF_ConstOp:$axis $axis_attr),
                  $keep_dims_0
                )
              ),
              $axis_1,
              $keep_dims_1
            )
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_SubOp:$sub
      (TF_ConstOp:$beta $beta_attr),
      (TF_MulOp:$mul_beta $mul_after_rsqrt_1, $mean_input_1)
    )
  ),
  (NativeCodeCall<"createLayerNormWithBody($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6, $7,"
                  " $8, $9, $10, $11, $12, $13, $14, $15, $16)"> $input, $gama, $beta, $epsilon_attr, 
   $axis_attr, $add_outer, $mul_input, $mul_after_rsqrt, $rsqrt, $add_mean, $const_epsilon, $mean_sqr, 
   $sqdiff, $mean_input, $axis, $sub, $mul_beta),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mul_after_rsqrt, $mul_after_rsqrt_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormSwapAdd : Pat<
  (TF_AddV2Op
    (TF_MulOp
      $input,
      (TF_MulOp:$mul_after_rsqrt
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                $input,
                (TF_MeanOp:$mean_input
                  $input,
                  (TF_ConstOp:$axis $axis_attr),
                  $keep_dims_0
                )
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_SubOp
      (TF_ConstOp:$beta $beta_attr),
      (TF_MulOp $mul_after_rsqrt_1, $mean_input_1)
    )
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mul_after_rsqrt, $mul_after_rsqrt_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormSwapMul : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_MulOp:$mul_after_rsqrt
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                $input,
                (TF_MeanOp:$mean_input
                  $input,
                  (TF_ConstOp:$axis $axis_attr),
                  $keep_dims_0
                )
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      ),
      $input
    ),
    (TF_SubOp
      (TF_ConstOp:$beta $beta_attr),
      (TF_MulOp $mul_after_rsqrt_1, $mean_input_1)
    )
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mul_after_rsqrt, $mul_after_rsqrt_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormSwapSquarediff : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_MulOp:$mul_after_rsqrt
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                (TF_MeanOp:$mean_input
                  $input,
                  (TF_ConstOp:$axis $axis_attr),
                  $keep_dims_0
                ),
                $input
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      ),
      $input
    ),
    (TF_SubOp
      (TF_ConstOp:$beta $beta_attr),
      (TF_MulOp $mul_after_rsqrt_1, $mean_input_1)
    )
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mul_after_rsqrt, $mul_after_rsqrt_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNorm_V2 : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_SubOp
        $input,
        (TF_MeanOp:$mean_input
          $input,
          (TF_ConstOp:$axis $axis_attr),
          $keep_dims_0
        )
      ),
      (TF_MulOp
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                $input,
                $mean_input_1
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_ConstOp:$beta $beta_attr)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormV3DisableMinimizeBrodcast : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_RsqrtOp
        (TF_AddV2Op
          (TF_MeanOp
            (TF_SquaredDifferenceOp
              $input,
              $mean_input_1
            ),
            $axis_1,
            $keep_dims_1
          ),
          (TF_ConstOp $epsilon_attr)
        )
      ),
      (TF_MulOp
        (TF_SubOp
          $input,
          (TF_MeanOp:$mean_input
            $input,
            (TF_ConstOp:$axis $axis_attr),
            $keep_dims_0
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_ConstOp:$beta $beta_attr)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormV4 : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_MulOp
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                $input,
                $mean_input_1
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_SubOp
          $input,
          (TF_MeanOp:$mean_input
            $input,
            (TF_ConstOp:$axis $axis_attr),
            $keep_dims_0
          )
        )
      ),
      (TF_ConstOp:$gama $gama_attr)
    ),
    (TF_ConstOp:$beta $beta_attr)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormV4SwapSquarediff : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_MulOp
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_MeanOp
              (TF_SquaredDifferenceOp
                $mean_input_1,
                $input
              ),
              $axis_1,
              $keep_dims_1
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_SubOp
          $input,
          (TF_MeanOp:$mean_input
            $input,
            (TF_ConstOp:$axis $axis_attr),
            $keep_dims_0
          )
        )
      ),
      (TF_ConstOp:$gama $gama_attr)
    ),
    (TF_ConstOp:$beta $beta_attr)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $mean_input, $mean_input_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormWithCast : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_SubOp
        $input,
        (TF_CastOp
          (TF_MeanOp:$input_cast_mean
            (TF_CastOp:$input_cast $input, $_),
            (TF_ConstOp:$axis $axis_attr),
            $keep_dims_0
          ),
          $_
        )
      ),
      (TF_MulOp
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_CastOp
              (TF_MeanOp
                (TF_SquaredDifferenceOp
                  $input_cast_1,
                  $input_cast_mean_1
                ),
                $axis_1,
                $keep_dims_1
              ),
              $_
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_ConstOp:$beta $beta_attr)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $input_cast, $input_cast_1),
   (SameTwoValuesOrAttrs $input_cast_mean, $input_cast_mean_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormWithCastV2 : Pat<
  (TF_AddV2Op
    (TF_MulOp
      $input,
      (TF_MulOp:$factor
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_CastOp
              (TF_MeanOp
                (TF_SquaredDifferenceOp
                  $input_cast_1,
                  (TF_MeanOp:$input_cast_mean
                    (TF_CastOp:$input_cast $input, $_),
                    (TF_ConstOp:$axis $axis_attr),
                    $keep_dims_0
                  )
                ),
                $axis_1,
                $keep_dims_1
              ),
              $_
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_ConstOp:$gama $gama_attr)
      )
    ),
    (TF_SubOp
      (TF_ConstOp:$beta $beta_attr),
      (TF_MulOp
        $factor_1,
        (TF_CastOp
          $input_cast_mean_1,
          $_
        )
      )
    )
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $input_cast, $input_cast_1),
   (SameTwoValuesOrAttrs $input_cast_mean, $input_cast_mean_1),
   (SameTwoValuesOrAttrs $factor, $factor_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;

def RewriteLayerNormWithCastDisableMinimizeBroadcast : Pat<
  (TF_AddV2Op
    (TF_MulOp
      (TF_MulOp
        (TF_RsqrtOp
          (TF_AddV2Op
            (TF_CastOp
              (TF_MeanOp
                (TF_SquaredDifferenceOp
                  $input_cast_1,
                  $input_cast_mean_1
                ),
                $axis_1,
                $keep_dims_1
              ),
              $_
            ),
            (TF_ConstOp $epsilon_attr)
          )
        ),
        (TF_SubOp
          $input,
          (TF_CastOp
            (TF_MeanOp:$input_cast_mean
              (TF_CastOp:$input_cast $input, $_),
              (TF_ConstOp:$axis $axis_attr),
              $keep_dims_0
            ),
            $_
          )
        )
      ),
      (TF_ConstOp:$gama $gama_attr)
    ),
    (TF_ConstOp:$beta $beta_attr)
  ),
  (NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4)"> $input, $gama, $beta, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims_0), (TrueBoolAttr $keep_dims_1),
   (SameTwoValuesOrAttrs $axis, $axis_1),
   (SameTwoValuesOrAttrs $input_cast, $input_cast_1),
   (SameTwoValuesOrAttrs $input_cast_mean, $input_cast_mean_1),
   (OneRank $gama_attr), (OneRank $beta_attr)]>;


//===----------------------------------------------------------------------===//
// L2Norm Pattern
//===----------------------------------------------------------------------===//

def RewriteL2NormV1 : Pat<
  (TF_MulOp
    $input,
    (TF_RsqrtOp
      (TF_MaximumOp
        (TF_SumOp
          (TF_SquareOp
            $input
          ),
          (TF_ConstOp:$axis $axis_attr),
          $keep_dims
        ),
        (TF_ConstOp $epsilon_attr)
      )
    )
  ),
  (NativeCodeCall<"createL2NormV1($_builder, $_loc, $0, $1, $2)"> $input, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims)]>;

def RewriteL2NormV1Multiplyer : Pat<
  (TF_MulOp
    $input,
    (TF_MulOp
      (TF_RsqrtOp
        (TF_MaximumOp
          (TF_SumOp
            (TF_SquareOp
              $input
            ),
            (TF_ConstOp:$axis $axis_attr),
            $keep_dims
          ),
          (TF_ConstOp $epsilon_attr)
        )
      ),
      (TF_ConstOp:$mulitplyer $mulitplyer_attr)
    )
  ),
  (NativeCodeCall<"createL2NormV1Multiplyer($_builder, $_loc, $0, $1, $2, $3)"> $input, $mulitplyer, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims)]>;
  
def RewriteL2NormV1SwapMul : Pat<
  (TF_MulOp
    (TF_RsqrtOp
      (TF_MaximumOp
        (TF_SumOp
          (TF_SquareOp
            $input
          ),
          (TF_ConstOp:$axis $axis_attr),
          $keep_dims
        ),
        (TF_ConstOp $epsilon_attr)
      )
    ),
    $input
  ),
  (NativeCodeCall<"createL2NormV1($_builder, $_loc, $0, $1, $2)"> $input, $epsilon_attr, $axis_attr),
  [(OneSize $epsilon_attr), (OneSize $axis_attr),
   (TrueBoolAttr $keep_dims)]>;

def RewriteL2NormV2 : Pat<
  (TF_MulOp
    (TF_RsqrtOp
      (TF_ReluOp
        (TF_SumOp
          (TF_SquareOp
            $input
          ),
          (TF_ConstOp:$axis $axis_attr),
          $keep_dims
        )
      )
    ),
    $input
  ),
  (NativeCodeCall<"createL2NormV2($_builder, $_loc, $0, $1)"> $input, $axis_attr),
  [(OneSize $axis_attr), (TrueBoolAttr $keep_dims)]>;

def RewriteL2NormV2SwapMul : Pat<
  (TF_MulOp
    $input,
    (TF_RsqrtOp
      (TF_ReluOp
        (TF_SumOp
          (TF_SquareOp
            $input
          ),
          (TF_ConstOp:$axis $axis_attr),
          $keep_dims
        )
      )
    )
  ),
  (NativeCodeCall<"createL2NormV2($_builder, $_loc, $0, $1)"> $input, $axis_attr),
  [(OneSize $axis_attr), (TrueBoolAttr $keep_dims)]>;

// note: x^2 guarantee >= 0, so set epsilon to 0.0
def RewriteL2NormV3 : Pat<
  (TF_MulOp
    $input,
    (TF_RsqrtOp
      (TF_SumOp
        (TF_SquareOp $input),
        (TF_ConstOp:$axis $axis_attr),
        $keep_dims
      )
    )
  ),
  (NativeCodeCall<"createL2NormV2($_builder, $_loc, $0, $1)"> $input, $axis_attr),
  [(OneSize $axis_attr), (TrueBoolAttr $keep_dims)]>;

//===----------------------------------------------------------------------===//
// GeLU Pattern
//===----------------------------------------------------------------------===//

def GeluValue0 : Constraint<CPred<"isSplatCloseToValue($0.dyn_cast<DenseFPElementsAttr>(), 0.5)">, "value attr is not splat value of 0.5">;
def GeluValue1 : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 1.0)">, "value attr is not splat value of 1.0">;
def GeluValue2 : Constraint<CPred<"isSplatCloseToValue($0.dyn_cast<DenseFPElementsAttr>(), 0.797884583, 0.0001)">, "value attr is not splat value of 0.797884583">;
def GeluValue3 : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 3.0)">, "value attr is not splat value of 3.0">;
def GeluValue4 : Constraint<CPred<"isSplatCloseToValue($0.dyn_cast<DenseFPElementsAttr>(), 4.471500e-02, 0.001)">, "value attr is not splat value of 4.471500e-02">;
def GeluValue5 : Constraint<CPred<"isSplatCloseToValue($0.dyn_cast<DenseFPElementsAttr>(), 0.707106769, 0.0001)">, "value attr is not splat value of 0.707106769">;

def RewriteGELUerf : Pat<
  (TF_MulOp
    (TF_AddV2Op
      (TF_ErfOp
        (TF_MulOp
          (TF_ConstOp $cst),
          $input
        )
      ),
      (TF_ConstOp $cst0)
    ),
    (TF_MulOp
      (TF_ConstOp $cst1),
      $input
    )
  ),
  (NativeCodeCall<"createGELU($_builder, $_loc, $0, \"erf\")"> $input),
  [(GeluValue0 $cst1), (GeluValue1 $cst0), (GeluValue5 $cst)]>;

def RewriteGELUerfV2 : Pat<
  (TF_MulOp
    $input,
    (TF_MulOp
      (TF_AddV2Op
        (TF_ErfOp
          (TF_MulOp
            $input,
            (TF_ConstOp $cst)
          )
        ),
        (TF_ConstOp $cst0)
      ),
      (TF_ConstOp $cst1)
    )
  ),
  (NativeCodeCall<"createGELU($_builder, $_loc, $0, \"erf\")"> $input),
  [(GeluValue0 $cst1), (GeluValue1 $cst0), (GeluValue5 $cst)]>;

def RewriteGELUtanh : Pat<
  (TF_MulOp
    (TF_AddV2Op
      (TF_TanhOp
        (TF_MulOp
          (TF_AddV2Op
            (TF_MulOp
              (TF_ConstOp $cst),
              (TF_PowOp
                $input,
                (TF_ConstOp $cst0)
              )
            ),
            $input
          ),
          (TF_ConstOp $cst3)
        )
      ),
      (TF_ConstOp $cst1)
    ),
    (TF_MulOp
      (TF_ConstOp $cst2),
      $input
    )
  ),
  (NativeCodeCall<"createGELU($_builder, $_loc, $0, \"tanh\")"> $input),
  [(GeluValue4 $cst), (GeluValue3 $cst0), (GeluValue1 $cst1), (GeluValue0 $cst2), (GeluValue2 $cst3)]>;


def RewriteGELUtanhV2 : Pat<
  (TF_MulOp
    $input,
    (TF_MulOp
      (TF_AddV2Op
        (TF_TanhOp
          (TF_MulOp
            (TF_AddV2Op
              $input,
              (TF_MulOp
                (TF_PowOp
                  $input,
                  (TF_ConstOp $cst0)
                ),
                (TF_ConstOp $cst)
              )
            ),
            (TF_ConstOp $cst3)
          )
        ),
        (TF_ConstOp $cst1)
      ),
      (TF_ConstOp $cst2)
    )
  ),
  (NativeCodeCall<"createGELU($_builder, $_loc, $0, \"tanh\")"> $input),
  [(GeluValue4 $cst), (GeluValue3 $cst0), (GeluValue1 $cst1), (GeluValue0 $cst2), (GeluValue2 $cst3)]>;


def RewriteGELUtanhV3 : Pat<
  (TF_MulOp
    (TF_AddV2Op
      (TF_TanhOp
        (TF_MulOp
          (TF_AddV2Op
            (TF_MulOp
              (TF_PowOp
                $input,
                (TF_ConstOp $cst4)
              ),
              (TF_ConstOp $cst3)
            ),
            $input
          ),
          (TF_ConstOp $cst2)
        )
      ),
      (TF_ConstOp $cst1)
    ),
    (TF_MulOp
      $input,
      (TF_ConstOp $cst0)
    )
  ),
  (NativeCodeCall<"createGELU($_builder, $_loc, $0, \"tanh\")"> $input),
  [(GeluValue0 $cst0), (GeluValue1 $cst1), (GeluValue2 $cst2), (GeluValue4 $cst3), (GeluValue3 $cst4)]>;


def RewriteGELUtanhV4 : Pat<
  (TF_MulOp
    (TF_AddV2Op
      (TF_TanhOp
        (TF_MulOp
          (TF_AddV2Op
            $input,
            (TF_MulOp
              (TF_PowOp
                $input,
                (TF_ConstOp $cst0)
              ),
              (TF_ConstOp $cst)
            )
          ),
          (TF_ConstOp $cst3)
        )
      ),
      (TF_ConstOp $cst1)
    ),
    (TF_MulOp
      $input,
      (TF_ConstOp $cst2)
    )
  ),
  (NativeCodeCall<"createGELU($_builder, $_loc, $0, \"tanh\")"> $input),
  [(GeluValue4 $cst), (GeluValue3 $cst0), (GeluValue1 $cst1), (GeluValue0 $cst2), (GeluValue2 $cst3)]>;


#endif // REWRITE_TO_CUSTOM_CALL_PATTERN
