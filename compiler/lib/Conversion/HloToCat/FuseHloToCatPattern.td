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
def ThreeSize : Constraint<CPred<"$0.size() == 3">, "three size">;
def FourSize : Constraint<CPred<"$0.size() == 4">, "four size">;
def OneRank : Constraint<CPred<"cast<ShapedType>($0.getType()).getRank() == 1">, "one rank">;
def TwoRank : Constraint<CPred<"cast<ShapedType>($0.getType()).getRank() == 2">, "two rank">;
def ThreeRank : Constraint<CPred<"cast<ShapedType>($0.getType()).getRank() == 3">, "three rank">;
def FourRank : Constraint<CPred<"cast<ShapedType>($0.getType()).getRank() == 4">, "four rank">;
def Permute10Check : Constraint<CPred<"$0.getValues<int64_t>()[0] == 1 && $0.getValues<int64_t>()[1] == 0">, "transpose <[1, 0]>">;
def Permute021Check : Constraint<CPred<"$0.getValues<int64_t>()[0] == 0 && $0.getValues<int64_t>()[1] == 2 && $0.getValues<int64_t>()[2] == 1">, "bmm 3d permute check (for transpose before bmm)">;
def Permute0213Check : Constraint<CPred<"$0.getValues<int64_t>()[0] == 0 && $0.getValues<int64_t>()[1] == 2 && $0.getValues<int64_t>()[2] == 1 && $0.getValues<int64_t>()[3] == 3">, "bmm 4d permute check (for transpose after bmm)">;
def TransposeCheck : Constraint<CPred<"$0.getValues<int64_t>()[0] == 1 && $0.getValues<int64_t>()[1] == 0">, "matrix transpose check">;
def BiasBroadcastDimCheck : Constraint<CPred<"$0.getValues<int64_t>()[0] == 1">, "bias broadcast dimension check">;
def LayoutRrrCheck : Constraint<CPred<"$0.getValue() == \"rrr\"">, "layout rrr check">;

def GemmBiasAddConstraint
    : Constraint<
          CPred<"cast<ShapedType>($0.getType()).getShape()[0] == cast<ShapedType>($1.getType()).getShape()[1]">,
          "bias add constraint">;

def IsGemmRCRDimNum
    : Constraint<
          CPred<"$0.getLhsContractingDimensions().size() == 1 && $0.getRhsContractingDimensions().size() == 1 && $0.getLhsContractingDimensions()[0] == 1 && $0.getRhsContractingDimensions()[0] == 1 && $0.getLhsBatchingDimensions().size() == 0 && $0.getLhsBatchingDimensions().size() == 0">,
          "is gemm rcr dimension">;

def MhloDotGeneralBiasToCatGemmRCRBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config),
            (MHLO_BroadcastInDimOp $bias, $dim)),
          (Cat_GemmRCRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (TwoRank $lhs), (TwoRank $rhs), (IsGemmRCRDimNum $dimension_numbers), (GemmBiasAddConstraint $bias, $output)]>;

def MhloBiasDotGeneralToCatGemmRCRBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_BroadcastInDimOp $bias, $dim),
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config)),
          (Cat_GemmRCRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (TwoRank $lhs), (TwoRank $rhs), (IsGemmRCRDimNum $dimension_numbers), (GemmBiasAddConstraint $bias, $output)]>;

def MhloAddCatGemmRCRBiasToCatGemmRCRBiasPattern
    : Pat<(MHLO_AddOp
            (Cat_GemmRCROp : $output
              $lhs, $rhs),
            (MHLO_BroadcastInDimOp $bias, $dim)),
          (Cat_GemmRCRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (TwoRank $lhs), (TwoRank $rhs), (GemmBiasAddConstraint $bias, $output)]>;

def MhloBiasCatGemmRCRAddToCatGemmRCRBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_BroadcastInDimOp $bias, $dim),
            (Cat_GemmRCROp : $output
              $lhs, $rhs)),
          (Cat_GemmRCRBiasOp $lhs, $rhs, $bias),
          [(OneRank $bias), (TwoRank $lhs), (TwoRank $rhs), (GemmBiasAddConstraint $bias, $output)]>;


// these patterns are commented because gemm_rrr_bias op is not supported by AIT
// def MhloBiasDotToCatGemmBiasPattern
//     : Pat<(MHLO_AddOp
//             (MHLO_BroadcastInDimOp $bias, $dim),
//             (MHLO_DotOp : $output
//               $lhs, $rhs, $precision_config)),
//           (Cat_GemmRRRBiasOp $lhs, $rhs, $bias),
//           [(OneRank $bias), (GemmBiasAddConstraint $bias, $output)]>;

// def MhloDotBiasToCatGemmBiasPattern
//     : Pat<(MHLO_AddOp
//             (MHLO_DotOp : $output
//               $lhs, $rhs, $precision_config),
//             (MHLO_BroadcastInDimOp $bias, $dim)),
//           (Cat_GemmRRRBiasOp $lhs, $rhs, $bias),
//           [(OneRank $bias), (GemmBiasAddConstraint $bias, $output)]>;

def getDim1Attr : NativeCodeCall<"$_builder.getI64IntegerAttr(cast<ShapedType>($0.getType()).getShape()[1])">;
def getDim2Attr : NativeCodeCall<"$_builder.getI64IntegerAttr(cast<ShapedType>($0.getType()).getShape()[2])">;
def GemmPermuteShapeCheck
    : Constraint<
        CPred<"cast<ShapedType>($0.getType()).getShape()[0] * cast<ShapedType>($0.getType()).getShape()[1] == cast<ShapedType>($1.getType()).getShape()[0] && cast<ShapedType>($0.getType()).getShape()[2] * cast<ShapedType>($0.getType()).getShape()[3] == cast<ShapedType>($1.getType()).getShape()[1]">, "gemm permute shape check">;

//===----------------------------------------------------------------------===//
// Mhlo Transpose Dot Transpose To Cat GemmRCRPermute Pattern
//===----------------------------------------------------------------------===//

def MhloCatGemmRCRTransToCatGemmRCRPermutePattern
    : Pat<(MHLO_TransposeOp
            (MHLO_ReshapeOp : $reshape_out
              (Cat_GemmRCROp : $gemm_out
                $lhs, $rhs)
            ),
            $permute),
          (Cat_GemmRCRPermuteOp $lhs, $rhs, (getDim1Attr $reshape_out), (getDim2Attr $reshape_out)),
          [(TwoRank $lhs), (TwoRank $rhs), (FourRank $reshape_out), (GemmPermuteShapeCheck $reshape_out, $gemm_out), (Permute0213Check $permute)]>;

def MhloCatGemmRRRTransToCatGemmRRRPermutePattern
    : Pat<(MHLO_TransposeOp
            (MHLO_ReshapeOp : $reshape_out
              (Cat_GemmRRROp : $gemm_out
                $lhs, $rhs)
            ),
            $permute),
          (Cat_GemmRRRPermuteOp $lhs, $rhs, (getDim1Attr $reshape_out), (getDim2Attr $reshape_out)),
          [(TwoRank $lhs), (TwoRank $rhs), (FourRank $reshape_out), (GemmPermuteShapeCheck $reshape_out, $gemm_out), (Permute0213Check $permute)]>;

def CatGemmRRRPermuteTransToCatGemmRCRPermutePattern
    : Pat<(Cat_GemmRRRPermuteOp
            $lhs,
            (MHLO_TransposeOp $rhs, $permute),
            $t1,
            $t2
          ),
          (Cat_GemmRCRPermuteOp $lhs, $rhs, $t1, $t2),
          [(TwoSize $permute), (Permute10Check $permute)]>;

def LayoutFrom3DDotGeneralDimNums : NativeCodeCall<"GetLayoutFrom3DDotGeneralDimNums($0, &$_builder)">;

def CheckRRRLayoutFrom3DDotGeneralDimNums
    : Constraint<
          CPred<"$0.getLhsContractingDimensions().size() == 1 && $0.getRhsContractingDimensions().size() == 1 && $0.getLhsContractingDimensions()[0] == 2 && $0.getRhsContractingDimensions()[0] == 1">,
          "is bmm rrr dimension">;


def CheckRCRLayoutFrom3DDotGeneralDimNums
    : Constraint<
          CPred<"$0.getLhsContractingDimensions().size() == 1 && $0.getRhsContractingDimensions().size() == 1 && $0.getLhsContractingDimensions()[0] == 2 && $0.getRhsContractingDimensions()[0] == 2">,
          "is bmm rcr dimension">;


def CheckCRRLayoutFrom3DDotGeneralDimNums
    : Constraint<
          CPred<"$0.getLhsContractingDimensions().size() == 1 && $0.getRhsContractingDimensions().size() == 1 && $0.getLhsContractingDimensions()[0] == 1 && $0.getRhsContractingDimensions()[0] == 1">,
          "is bmm crr dimension">;

def CheckCCRLayoutFrom3DDotGeneralDimNums
    : Constraint<
          CPred<"$0.getLhsContractingDimensions().size() == 1 && $0.getRhsContractingDimensions().size() == 1 && $0.getLhsContractingDimensions()[0] == 1 && $0.getRhsContractingDimensions()[0] == 2">,
          "is bmm ccr dimension">;

def CheckBMMPermuteShapeSplitOnBatch
    : Constraint<
          CPred<"cast<ShapedType>($0.getType()).getShape()[0] * cast<ShapedType>($0.getType()).getShape()[1] == cast<ShapedType>($1.getType()).getShape()[0] && cast<ShapedType>($0.getType()).getShape()[2] == cast<ShapedType>($1.getType()).getShape()[1]">,
          "bmm rrr Shape Split On Batch">;


def MhloDotGeneralReshapeTransposeToBMMRRRPermutePattern
    : Pat<(MHLO_TransposeOp
            (MHLO_ReshapeOp : $reshape_out
              (MHLO_DotGeneralOp : $bmm_out
                $lhs, $rhs, $dimension_numbers, $precision_config)
            ),
            $permute),
          (Cat_BMMRRRPermuteOp $lhs, $rhs, (getDim1Attr $reshape_out)),
          [
            (ThreeRank $lhs), 
            (ThreeRank $rhs), 
            (FourSize $permute), 
            (Permute0213Check $permute),
            (CheckRRRLayoutFrom3DDotGeneralDimNums $dimension_numbers),
            (CheckBMMPermuteShapeSplitOnBatch $reshape_out, $lhs)
          ]>;

def MhloDotGeneralReshapeTransposeToBMMRCRPermutePattern
    : Pat<(Cat_BMMRRRPermuteOp 
            $lhs, 
            (MHLO_TransposeOp $transpose_in, $permute), 
            $shape),
          (Cat_BMMRCRPermuteOp $lhs, $transpose_in, $shape),
          [
            (ThreeRank $lhs), 
            (ThreeRank $transpose_in), 
            (ThreeSize $permute), 
            (Permute021Check $permute)
          ]>;

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
            (MHLO_DotOp $lhs, $rhs, $precision),
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

def MhloDotAddBroadcastToGemmRRRBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_BroadcastInDimOp $bias, $dim),
            (MHLO_DotOp $lhs, $rhs, $precision)),
          (Cat_GemmRRRBiasOp 
            $lhs, 
            $rhs, 
            $bias),
          [ (TwoRank $lhs), 
            (TwoRank $rhs),
            (OneRank $bias),
            (OneSize $dim),
            (BiasBroadcastDimCheck $dim) ]>;

def CatGemmRRRBroadcastAddToGemmRRRBiasPattern
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

def CatGemmRRRAddBroadcastToGemmRRRBiasPattern
    : Pat<(MHLO_AddOp
            (MHLO_BroadcastInDimOp $bias, $dim),
            (Cat_GemmRRROp $lhs, $rhs)),
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

def MhloTransposeDotBroadcastAddToGemmRCRBiasPattern
    : Pat<(MHLO_AddOp
            (Cat_GemmRRROp
              $lhs,
              (MHLO_TransposeOp
                      $transpose_in, $permute)
            ),
            (MHLO_BroadcastInDimOp $bias, $dim)),
          (Cat_GemmRCRBiasOp $lhs, $transpose_in, $bias),
          [
            (TwoRank $transpose_in), 
            (TwoRank $lhs), 
            (TwoSize $permute),
            (OneRank $bias), 
            (TransposeCheck $permute),
            (OneSize $dim),
            (BiasBroadcastDimCheck $dim)
          ]>;

def MhloDotGeneralAddToBMMRRRAddPattern
    : Pat<(MHLO_AddOp
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config),
            $add),
          (Cat_BMMRRRAddOp $lhs, $rhs, $add),
          [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckRRRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

def MhloAddDotGeneralToBMMRRRAddPattern
    : Pat<(MHLO_AddOp
            $add,
            (MHLO_DotGeneralOp : $output
              $lhs, $rhs, $dimension_numbers, $precision_config)),
          (Cat_BMMRRRAddOp $lhs, $rhs, $add),
          [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckRRRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

// def MhloDotGeneralAddToBMMRCRAddPattern
//     : Pat<(MHLO_AddOp
//             (MHLO_DotGeneralOp : $output
//               $lhs, $rhs, $dimension_numbers, $precision_config),
//             $add),
//           (Cat_BMMRCRAddOp $lhs, $rhs, $add),
//           [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckRCRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

// def MhloAddDotGeneralToBMMRCRAddPattern
//     : Pat<(MHLO_AddOp
//             $add,
//             (MHLO_DotGeneralOp : $output
//               $lhs, $rhs, $dimension_numbers, $precision_config)),
//           (Cat_BMMRCRAddOp $lhs, $rhs, $add),
//           [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckRCRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

// def MhloDotGeneralAddToBMMCRRAddPattern
//     : Pat<(MHLO_AddOp
//             (MHLO_DotGeneralOp : $output
//               $lhs, $rhs, $dimension_numbers, $precision_config),
//             $add),
//           (Cat_BMMCRRAddOp $lhs, $rhs, $add),
//           [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckCRRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

// def MhloAddDotGeneralToBMMCRRAddPattern
//     : Pat<(MHLO_AddOp
//             $add,
//             (MHLO_DotGeneralOp : $output
//               $lhs, $rhs, $dimension_numbers, $precision_config)),
//           (Cat_BMMCRRAddOp $lhs, $rhs, $add),
//           [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckCRRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

// def MhloDotGeneralAddToBMMCCRAddPattern
//     : Pat<(MHLO_AddOp
//             (MHLO_DotGeneralOp : $output
//               $lhs, $rhs, $dimension_numbers, $precision_config),
//             $add),
//           (Cat_BMMCCRAddOp $lhs, $rhs, $add),
//           [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckCCRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

// def MhloAddDotGeneralToBMMCCRAddPattern
//     : Pat<(MHLO_AddOp
//             $add,
//             (MHLO_DotGeneralOp : $output
//               $lhs, $rhs, $dimension_numbers, $precision_config)),
//           (Cat_BMMCCRAddOp $lhs, $rhs, $add),
//           [(ThreeRank $lhs), (ThreeRank $rhs), (ThreeRank $add), (CheckCCRLayoutFrom3DDotGeneralDimNums $dimension_numbers)]>;

def MhloTransposeBMMRRRAddToBMMCRRAddPattern
    : Pat<(Cat_BMMRRRAddOp 
            (MHLO_TransposeOp $transpose_in, $permute),
            $rhs, 
            $add),
          (Cat_BMMCRRAddOp $transpose_in, $rhs, $add),
          [(ThreeRank $transpose_in), (ThreeRank $rhs), (ThreeRank $add), (Permute021Check $permute)]>;


//===----------------------------------------------------------------------===//
// Cat GemmRRRBias To Cat GemmRCRBias Pattern
//===----------------------------------------------------------------------===//

def MhloTransposeGemmRRRBiasToGemmRCRBiasPattern
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

// support only fuse nhwc layout
def ConvBiasAddConstraint
    : Constraint<
          CPred<"*$0.begin() == cast<ShapedType>($1.getType()).getRank() - 1">,
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

//===----------------------------------------------------------------------===//
// Mhlo 3D Dot Genral To Cat BMM XXX (layout) Pattern
//===----------------------------------------------------------------------===//

def Mhlo3DDotGeneralToBMMRRRPattern
    : Pat<(MHLO_DotGeneralOp : $bmm_out
            $lhs, $rhs, $dimension_numbers, $precision_config),
          (Cat_BMMRRROp $lhs, $rhs),
          [
            (ThreeRank $lhs), 
            (ThreeRank $rhs), 
            (CheckRRRLayoutFrom3DDotGeneralDimNums $dimension_numbers)
          ]>;

def Mhlo3DDotGeneralToBMMRCRPattern
    : Pat<(Cat_BMMRRROp 
            $lhs, 
            (MHLO_TransposeOp $transpose_in, $permute)),
          (Cat_BMMRCROp $lhs, $transpose_in),
          [
            (ThreeRank $lhs), 
            (ThreeRank $transpose_in), 
            (Permute021Check $permute)
          ]>;

def Mhlo3DDotGeneralToBMMCRRPattern
    : Pat<(Cat_BMMRRROp 
            (MHLO_TransposeOp $transpose_in, $permute),
            $rhs
            ),
          (Cat_BMMCRROp $transpose_in, $rhs),
          [
            (ThreeRank $rhs), 
            (ThreeRank $transpose_in), 
            (Permute021Check $permute)
          ]>;   

def CatBMMCRRToBMMCRCPattern
    : Pat<(MHLO_TransposeOp 
            (Cat_BMMCRROp $lhs, $rhs), 
            $permute
          ),
          (Cat_BMMCRCOp $lhs, $rhs),
          [
            (ThreeRank $lhs), 
            (ThreeRank $rhs), 
            (Permute021Check $permute)
          ]>;

def MhloTransposeCatBMMRRRToBMMRRCPattern
    : Pat<(MHLO_TransposeOp 
            (Cat_BMMRRROp $lhs, $rhs), 
            $permute
          ),
          (Cat_BMMRRCOp $lhs, $rhs),
          [
            (ThreeRank $lhs), 
            (ThreeRank $rhs), 
            (Permute021Check $permute)
          ]>;

#endif // BYTEIR_CONVERSION_HLOTOCAT_FUSE_MHLO_TO_CAT_PATTERN
