//===- MhloToAceActivationPattern.td --------------------------------------===//
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

#ifndef BYTEIR_CONVERSION_HLOTOCAT_FUSE_MHLO_TO_CAT_PATTERN
#define BYTEIR_CONVERSION_HLOTOCAT_FUSE_MHLO_TO_CAT_PATTERN

include "byteir/Dialect/Cat/IR/CatOps.td"
include "mhlo/IR/hlo_ops.td"
include "mlir/IR/OpBase.td"

def OneSize : Constraint<CPred<"$0.size() == 1">, "one size">;
def TwoSize : Constraint<CPred<"$0.size() == 2">, "two size">;
def FourSize : Constraint<CPred<"$0.size() == 4">, "four size">;
def OneRank : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 1">, "one rank">;
def TwoRank : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 2">, "two rank">;
def ThreeRank : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 3">, "three rank">;
def FourRank : Constraint<CPred<"$0.getType().cast<ShapedType>().getRank() == 4">, "four rank">;
def BMMPermuteCheck : Constraint<CPred<"$0.getValues<int64_t>()[0] == 0 && $0.getValues<int64_t>()[1] == 2 && $0.getValues<int64_t>()[2] == 1 && $0.getValues<int64_t>()[3] == 3">, "bmm permute check">;
def TransposeCheck : Constraint<CPred<"$0.getValues<int64_t>()[0] == 1 && $0.getValues<int64_t>()[1] == 0">, "matrix transpose check">;
def BiasBroadcastDimCheck : Constraint<CPred<"$0.getValues<int64_t>()[0] == 1">, "bias broadcast dimension check">;
def LayoutRrrCheck : Constraint<CPred<"$0.getValue() == \"rrr\"">, "layout rrr check">;

def GemmBiasAddConstraint
    : Constraint<
          CPred<"$0.getType().cast<ShapedType>().getShape()[0] == $1.getType().cast<ShapedType>().getShape()[1]">,
          "bias add constraint">;

def IsGemmRCRDimNum
    : Constraint<
          CPred<"$0.getLhsContractingDimensions().size() == 1 && $0.getRhsContractingDimensions().size() == 1 && $0.getLhsContractingDimensions()[0] == 1 && $0.getRhsContractingDimensions()[0] == 1 && $0.getLhsBatchingDimensions().size() == 0 && $0.getLhsBatchingDimensions().size() == 0">,
          "is gemm rcr dimension">;

def MhloDotGeneralBiasToCatGemmBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config),
            (MHLO_BroadcastInDimOp $bias, $dim)),
          (Cat_GemmRCRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (TwoRank $lhs), (TwoRank $rhs), (IsGemmRCRDimNum $dimension_numbers), (GemmBiasAddConstraint $bias, $output)]>;

def MhloBiasDotGeneralToCatGemmBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_BroadcastInDimOp $bias, $dim),
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config)),
          (Cat_GemmRCRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (TwoRank $lhs), (TwoRank $rhs), (IsGemmRCRDimNum $dimension_numbers), (GemmBiasAddConstraint $bias, $output)]>;

def MhloBiasDotToCatGemmBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_BroadcastInDimOp $bias, $dim),
            (MHLO_DotOp : $output
              $lhs, $rhs, $precision_config)),
          (Cat_GemmRRRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (GemmBiasAddConstraint $bias, $output)]>;

def MhloDotBiasToCatGemmBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_DotOp : $output
              $lhs, $rhs, $precision_config),
            (MHLO_BroadcastInDimOp $bias, $dim)),
          (Cat_GemmRRRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (GemmBiasAddConstraint $bias, $output)]>;

def LayoutFrom3DDotGeneralDimNums : NativeCodeCall<"GetLayoutFrom3DDotGeneralDimNums($0, &$_builder)">;

def getBMMShapeDAttr : NativeCodeCall<"$_builder.getI64IntegerAttr($0.getType().cast<ShapedType>().getShape()[1])">;

def MhloDotGeneralReshapeTransposeToBMMPermutePattern
    : Pat<(MHLO_TransposeOp
            (MHLO_ReshapeOp : $reshape_out
              (MHLO_DotGeneralOp : $bmm_out
                $lhs, $rhs, $dimension_numbers, $precision_config)
            ),
            $permute),
          (Cat_BMMPermuteOp $lhs, $rhs, (getBMMShapeDAttr $reshape_out),
              (LayoutFrom3DDotGeneralDimNums $dimension_numbers)),
          [(ThreeRank $lhs), (ThreeRank $rhs), (FourSize $permute), (BMMPermuteCheck $permute)]>;

//===----------------------------------------------------------------------===//
// Mhlo Dot To Cat GemmRRR Pattern
//===----------------------------------------------------------------------===//

def MhloDotToGemmRRRPattern
    : Pat<(MHLO_DotOp
            $lhs,
            $rhs,
            $precision
          ),
          (Cat_GemmRRROp $lhs, $rhs),
          [(TwoRank $rhs), (TwoRank $lhs) ]>;

//===----------------------------------------------------------------------===//
// Mhlo Transpose Dot To Cat GemmRCR Pattern
//===----------------------------------------------------------------------===//

def MhloTransposeDotToGemmRCRPattern
    : Pat<(Cat_GemmRRROp
            $lhs,
            (MHLO_TransposeOp
                    $transpose_in, $permute)
          ),
          (Cat_GemmRCROp $lhs, $transpose_in),
          [
            (TwoRank $transpose_in), 
            (TwoRank $lhs), 
            (TwoSize $permute), 
            (TransposeCheck $permute)
          ]>;

//===----------------------------------------------------------------------===//
// Mhlo Dot Broadcast Add To Cat GemmRRRBias Pattern
//===----------------------------------------------------------------------===//

def MhloDotBroadcastAddToGemmRRRBiasPattern
    : Pat<(MHLO_AddOp
            (Cat_GemmRRROp $lhs, $rhs),
            (MHLO_BroadcastInDimOp $bias, $dim)),
          (Cat_GemmRRRBiasOp 
            $lhs, 
            $rhs, 
            $bias),
          [ (TwoRank $lhs), 
            (TwoRank $rhs),
            (OneRank $bias),
            (OneSize $dim),
            (BiasBroadcastDimCheck $dim) ]>;

//===----------------------------------------------------------------------===//
// Mhlo Transpose Dot Broadcast Add To Cat GemmRCRBias Pattern
//===----------------------------------------------------------------------===//

def MhloTransposeDotBroadcastAddToGemmPattern
    : Pat<(Cat_GemmRRRBiasOp
            $lhs,
            (MHLO_TransposeOp
                    $transpose_in, $permute),
            $bias
          ),
          (Cat_GemmRCRBiasOp $lhs, $transpose_in, $bias),
          [
            (TwoRank $transpose_in), 
            (TwoRank $lhs), 
            (TwoSize $permute),
            (OneRank $bias), 
            (TransposeCheck $permute)
          ]>;

def MhloDotGeneralAddToBMMAddPattern
    : Pat<(MHLO_AddOp
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config),
            $add),
          (Cat_BMMAddOp $lhs, $rhs, $add, (LayoutFrom3DDotGeneralDimNums $dimension_numbers)),
          [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add)]>;

def MhloAddDotGeneralToBMMAddPattern
    : Pat<(MHLO_AddOp
            $add,
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config)),
          (Cat_BMMAddOp $lhs, $rhs, $add, (LayoutFrom3DDotGeneralDimNums $dimension_numbers)),
          [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add)]>;

// support only fuse nhwc layout
def ConvBiasAddConstraint
    : Constraint<
          CPred<"*$0.begin() == $1.getType().cast<ShapedType>().getRank() - 1">,
          "bias add constraint">;

def LayoutFromConvDimNums : NativeCodeCall<"GetLayoutFromConvDimNums($0, &$_builder)">;

def MhloConvBroadcastAddToCatConvBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_ConvolutionOp : $output
              $lhs, $rhs,
              $window_strides, $padding, $lhs_dilation, $rhs_dilation,
              $window_reversal, $dimension_numbers, $feature_group_count, $batch_group_count, $precision_config),
            (MHLO_BroadcastInDimOp(MHLO_ConstantOp : $bias $_),
              $dim)),
          (Cat_Conv2dBiasOp $lhs, $rhs, $bias, (LayoutFromConvDimNums $dimension_numbers), 
            $window_strides, $padding, $lhs_dilation, $rhs_dilation),
          [(OneSize $dim), (FourRank $lhs), (FourRank $rhs), (ConvBiasAddConstraint $dim, $output)]>;

def AttrIsZero : Constraint<CPred<"mlir::isZeroAttribute($_self)">>;

def MhloConvBroadcastAddMaxToCatConvBiasReluPattern
    : Pat<(MHLO_MaxOp
            (MHLO_AddOp
              (MHLO_ConvolutionOp : $output
                $lhs, $rhs,
                $window_strides, $padding, $lhs_dilation, $rhs_dilation,
                $window_reversal, $dimension_numbers, $feature_group_count, $batch_group_count, $precision_config),
              (MHLO_BroadcastInDimOp(MHLO_ConstantOp : $bias $value),
                $dim)),
            (MHLO_ConstantOp : $_ $th)),
          (Cat_Conv2dBiasReluOp $lhs, $rhs, $bias, (LayoutFromConvDimNums $dimension_numbers), 
            $window_strides, $padding, $lhs_dilation, $rhs_dilation),
          [(OneSize $dim), (FourRank $lhs), (FourRank $rhs), (ConvBiasAddConstraint $dim, $output), (AttrIsZero:$th)]>;

def MhloConvBroadcastAddAddMaxToCatConvBiasAddReluPattern
    : Pat<(MHLO_MaxOp
            (MHLO_AddOp
              (MHLO_AddOp
                (MHLO_ConvolutionOp : $output
                  $lhs, $rhs,
                  $window_strides, $padding, $lhs_dilation, $rhs_dilation,
                  $window_reversal, $dimension_numbers, $feature_group_count, $batch_group_count, $precision_config),
                (MHLO_BroadcastInDimOp(MHLO_ConstantOp : $bias $value),
                  $dim)),
              $res),
            (MHLO_ConstantOp : $_ $th)),
          (Cat_Conv2dBiasAddReluOp $lhs, $rhs, $bias, $res, (LayoutFromConvDimNums $dimension_numbers), 
            $window_strides, $padding, $lhs_dilation, $rhs_dilation),
          [(OneSize $dim), (FourRank $lhs), (FourRank $rhs), (ConvBiasAddConstraint $dim, $output), (AttrIsZero:$th)]>;

#endif // BYTEIR_CONVERSION_HLOTOCAT_FUSE_MHLO_TO_CAT_PATTERN
