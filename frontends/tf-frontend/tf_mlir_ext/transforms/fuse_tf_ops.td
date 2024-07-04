//===- fuse_tf_ops.td ------------------------------------*--- tablegen -*-===//
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

#ifndef FUSE_TF_OPS_PATTERN
#define FUSE_TF_OPS_PATTERN

include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mhlo/IR/hlo_ops.td"

//===----------------------------------------------------------------------===//
// Sigmoid Pattern
//===----------------------------------------------------------------------===//

def FpSplatValueZero : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 0.0)">, "value attr is not splat value of 0.0">;
def FpSplatValueOne : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 1.0)">, "value attr is not splat value of 1.0">;
def IntSplatValueZero : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), 0)">, "value attr is not splat value of 0.0">;
def IntSplatValueOne : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), 1)">, "value attr is not splat value of 1.0">;
def IntSplatValueNegOne : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), -1)">, "value attr is not splat value of -1">;
def WhereValue0 : Constraint<CPred<"isSplatCloseToValue($0.dyn_cast<DenseFPElementsAttr>(), 0.0000115633, 0.000001)">, "value attr is not splat value of 0.0000115633">;
def WhereValue1 : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), 24)">, "value attr is not splat value of 24">;
def WhereValue2 : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), 28800)">, "value attr is not splat value of 28800">;
def WhereValue3 : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseIntElementsAttr>(), 86400)">, "value attr is not splat value of 86400">;
def WhereValue4 : Constraint<CPred<"isSplatValue($0.dyn_cast<DenseFPElementsAttr>(), 24.0)">, "value attr is not splat value of 24.0">;
def AxisAttrNegOne : Constraint<CPred<"$0.dyn_cast<IntegerAttr>().getValue() == -1">, "axis attr value is not -1">;
def AxisAttrZero : Constraint<CPred<"$0.dyn_cast<IntegerAttr>().getValue() == 0">, "axis attr value is not 0">;
def AxisAttrOne : Constraint<CPred<"$0.dyn_cast<IntegerAttr>().getValue() == 1">, "axis attr value is not 1">;
def OneRank : Constraint<CPred<"$0.getType().dyn_cast<ShapedType>().getRank() == 1">, "rank is not 1">;
def TwoRank : Constraint<CPred<"$0.getType().dyn_cast<ShapedType>().getRank() == 2">, "rank is not 2">;
def IntegerEleType : Constraint<CPred<"isa<IntegerType>($0.getType().dyn_cast<ShapedType>().getElementType())">, "type is not integer type">;
def FloatEleType : Constraint<CPred<"isa<FloatType>($0.getType().dyn_cast<ShapedType>().getElementType())">, "type is not float type">;
def SameEleType : Constraint<CPred<"$0.getType().dyn_cast<RankedTensorType>().getElementType() == $1.getType().dyn_cast<RankedTensorType>().getElementType()">, "type not equal">;

def FuseSigmoid : Pat<
  (TF_ReciprocalOp
    (TF_AddV2Op
      (TF_ExpOp
        (TF_NegOp $input)
      ),
      (TF_ConstOp $one)
    )
  ),
  (TF_SigmoidOp $input),
  [(FpSplatValueOne $one)]>;

def ReplaceWhereDynamic : Pat<
  (TF_GatherV2Op
    (TF_ReshapeOp:$after_reshape1
      $input1,
      $input_shape
    ),
    (TF_SqueezeOp:$after_squeeze1
      (TF_WhereOp:$after_where
        (TF_CastOp:$after_cast2
          (TF_ReshapeOp:$before_cast2
            (TF_OneHotOp:$after_onehot
              (TF_SqueezeOp:$after_squeeze
                (TF_CastOp:$after_cast1
                  (TF_MulOp:$before_cast1
                    (TF_FloorModOp
                      (TF_MulOp
                        (TF_CastOp:$after_cast
                          (TF_FloorModOp:$before_cast
                            (TF_AddV2Op
                              $input,
                              (TF_ConstOp:$addconst $addconst_attr)
                            ),
                            (TF_ConstOp:$floorconst $floorconst_attr)
                          ),
                          $truncate
                        ),
                        (TF_ConstOp:$mulconst $mulconst_attr)
                      ),
                      (TF_ConstOp:$floorconst1 $floorconst1_attr)
                    ),
                    (TF_ConstOp:$mulconst1 $mulconst1_attr)
                  ),
                  $truncate1
                ),
                $squeeze_dims
              ),
              (TF_ConstOp:$depth $depth_attr),
              (TF_ConstOp:$onvalue $onvalue_attr),
              (TF_ConstOp:$offvalue $offvalue_attr),
              $onehot_axis
            ),
            (TF_ConstOp:$shape $shape_attr)
          ),
          $truncate2
        )
      ),
      $squeeze_dims1
    ),
    (TF_ConstOp:$gatheraxis $gatheraxis_attr),
    $gather_batch_dims
  ),
  (NativeCodeCall<"replaceWhereDynamic($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6)"> $input1, $after_squeeze, $depth, $onvalue, $offvalue, $gatheraxis, $onehot_axis),
  [(WhereValue2 $addconst_attr),  (WhereValue3 $floorconst_attr),
   (WhereValue0 $mulconst_attr),  (FpSplatValueOne $floorconst1_attr),
   (WhereValue4 $mulconst1_attr), (WhereValue1 $depth_attr),
   (FpSplatValueOne $onvalue_attr), (FpSplatValueZero $offvalue_attr),
   (IntSplatValueNegOne $shape_attr), (IntSplatValueZero $gatheraxis_attr),
   (AxisAttrNegOne $onehot_axis), (AxisAttrZero $gather_batch_dims),
   (IntegerEleType $before_cast), (FloatEleType $after_cast),
   (FloatEleType $before_cast1), (IntegerEleType $after_cast1),
   (FloatEleType $before_cast2), (FloatEleType $after_cast2),
   (TwoRank $after_cast1), (OneRank $after_squeeze),
   (TwoRank $after_onehot), (OneRank $before_cast2),
   (TwoRank $after_where), (OneRank $after_squeeze1),
   (SameEleType $input1, $after_onehot)]>;

def ReplaceWhereStatic : Pat<
  (TF_GatherV2Op
    (TF_ReshapeOp:$after_reshape1
      $input1,
      $input_shape
    ),
    (TF_SqueezeOp:$after_squeeze1
      (TF_WhereOp:$after_where
        (TF_CastOp:$after_cast2
          (TF_ReshapeOp:$before_cast2
            (TF_OneHotOp:$after_onehot
              (TF_SqueezeOp:$after_squeeze
                (TF_CastOp:$after_cast1
                  (TF_MulOp:$before_cast1
                    (TF_FloorModOp
                      (TF_MulOp
                        (TF_CastOp:$after_cast
                          (TF_FloorModOp:$before_cast
                            (TF_AddV2Op
                              $input,
                              (TF_ConstOp:$addconst $addconst_attr)
                            ),
                            (TF_ConstOp:$floorconst $floorconst_attr)
                          ),
                          $truncate
                        ),
                        (TF_ConstOp:$mulconst $mulconst_attr)
                      ),
                      (TF_ConstOp:$floorconst1 $floorconst1_attr)
                    ),
                    (TF_ConstOp:$mulconst1 $mulconst1_attr)
                  ),
                  $truncate1
                ),
                $squeeze_dims
              ),
              (TF_ConstOp:$depth $depth_attr),
              (TF_ConstOp:$onvalue $onvalue_attr),
              (TF_ConstOp:$offvalue $offvalue_attr),
              $onehot_axis
            ),
            (TF_ConstOp:$shape $shape_attr)
          ),
          $truncate2
        )
      ),
      $squeeze_dims1
    ),
    (TF_ConstOp:$gatheraxis $gatheraxis_attr),
    $gather_batch_dims
  ),
  (NativeCodeCall<"replaceWhereStatic($_builder, $_loc, $0, $1)"> $input1, $after_onehot),
  [(WhereValue2 $addconst_attr),  (WhereValue3 $floorconst_attr),
   (WhereValue0 $mulconst_attr),  (FpSplatValueOne $floorconst1_attr),
   (WhereValue4 $mulconst1_attr), (WhereValue1 $depth_attr),
   (FpSplatValueOne $onvalue_attr), (FpSplatValueZero $offvalue_attr),
   (IntSplatValueNegOne $shape_attr), (IntSplatValueZero $gatheraxis_attr),
   (AxisAttrNegOne $onehot_axis), (AxisAttrZero $gather_batch_dims),
   (IntegerEleType $before_cast), (FloatEleType $after_cast),
   (FloatEleType $before_cast1), (IntegerEleType $after_cast1),
   (FloatEleType $before_cast2), (FloatEleType $after_cast2),
   (TwoRank $after_cast1), (OneRank $after_squeeze),
   (TwoRank $after_onehot), (OneRank $before_cast2),
   (TwoRank $after_where), (OneRank $after_squeeze1),
   (SameEleType $input1, $after_onehot)]>;

def ReplaceWhereDynamicV2 : Pat<
  (TF_GatherV2Op
    (TF_ReshapeOp:$after_reshape1
      $input1,
      $input_shape
    ),
    (TF_SqueezeOp:$after_squeeze1
      (TF_WhereOp:$after_where
        (TF_CastOp:$after_cast2
          (TF_ReshapeOp:$before_cast2
            (TF_OneHotOp:$after_onehot
              (TF_SqueezeOp:$after_squeeze
                (TF_CastOp:$after_cast1
                  (TF_MulOp:$before_cast1
                    (TF_FloorModOp
                      (TF_CastOp
                        (TF_MulOp
                          (TF_CastOp
                            (TF_CastOp:$after_cast
                              (TF_FloorModOp:$before_cast
                                (TF_AddV2Op
                                  $input,
                                  (TF_ConstOp:$addconst $addconst_attr)
                                ),
                                (TF_ConstOp:$floorconst $floorconst_attr)
                              ),
                              $truncate
                            ),
                            $truncate1
                          ),
                          (TF_ConstOp:$mulconst $mulconst_attr)
                        ),
                        $truncate2
                      ),
                      (TF_ConstOp:$floorconst1 $floorconst1_attr)
                    ),
                    (TF_ConstOp:$mulconst1 $mulconst1_attr)
                  ),
                  $truncate3
                ),
                $squeeze_dims
              ),
              (TF_ConstOp:$depth $depth_attr),
              (TF_ConstOp:$onvalue $onvalue_attr),
              (TF_ConstOp:$offvalue $offvalue_attr),
              $onehot_axis
            ),
            (TF_ConstOp:$shape $shape_attr)
          ),
          $truncate4
        )
      ),
      $squeeze_dims1
    ),
    (TF_ConstOp:$gatheraxis $gatheraxis_attr),
    $gather_batch_dims
  ),
  (NativeCodeCall<"replaceWhereDynamic($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6)"> $input1, $after_squeeze, $depth, $onvalue, $offvalue, $gatheraxis, $onehot_axis),
  [(WhereValue2 $addconst_attr),  (WhereValue3 $floorconst_attr),
   (WhereValue0 $mulconst_attr),  (FpSplatValueOne $floorconst1_attr),
   (WhereValue4 $mulconst1_attr), (WhereValue1 $depth_attr),
   (FpSplatValueOne $onvalue_attr), (FpSplatValueZero $offvalue_attr),
   (IntSplatValueNegOne $shape_attr), (IntSplatValueZero $gatheraxis_attr),
   (AxisAttrNegOne $onehot_axis), (AxisAttrZero $gather_batch_dims),
   (IntegerEleType $before_cast), (FloatEleType $after_cast),
   (FloatEleType $before_cast1), (IntegerEleType $after_cast1),
   (FloatEleType $before_cast2), (FloatEleType $after_cast2),
   (TwoRank $after_cast1), (OneRank $after_squeeze),
   (TwoRank $after_onehot), (OneRank $before_cast2),
   (TwoRank $after_where), (OneRank $after_squeeze1),
   (SameEleType $input1, $after_onehot)]>;

def ReplaceWhereStaticV2 : Pat<
  (TF_GatherV2Op
    (TF_ReshapeOp:$after_reshape1
      $input1,
      $input_shape
    ),
    (TF_SqueezeOp:$after_squeeze1
      (TF_WhereOp:$after_where
        (TF_CastOp:$after_cast2
          (TF_ReshapeOp:$before_cast2
            (TF_OneHotOp:$after_onehot
              (TF_SqueezeOp:$after_squeeze
                (TF_CastOp:$after_cast1
                  (TF_MulOp:$before_cast1
                    (TF_FloorModOp
                      (TF_CastOp
                        (TF_MulOp
                          (TF_CastOp
                            (TF_CastOp:$after_cast
                              (TF_FloorModOp:$before_cast
                                (TF_AddV2Op
                                  $input,
                                  (TF_ConstOp:$addconst $addconst_attr)
                                ),
                                (TF_ConstOp:$floorconst $floorconst_attr)
                              ),
                              $truncate
                            ),
                            $truncate1
                          ),
                          (TF_ConstOp:$mulconst $mulconst_attr)
                        ),
                        $truncate2
                      ),
                      (TF_ConstOp:$floorconst1 $floorconst1_attr)
                    ),
                    (TF_ConstOp:$mulconst1 $mulconst1_attr)
                  ),
                  $truncate3
                ),
                $squeeze_dims
              ),
              (TF_ConstOp:$depth $depth_attr),
              (TF_ConstOp:$onvalue $onvalue_attr),
              (TF_ConstOp:$offvalue $offvalue_attr),
              $onehot_axis
            ),
            (TF_ConstOp:$shape $shape_attr)
          ),
          $truncate4
        )
      ),
      $squeeze_dims1
    ),
    (TF_ConstOp:$gatheraxis $gatheraxis_attr),
    $gather_batch_dims
  ),
  (NativeCodeCall<"replaceWhereStatic($_builder, $_loc, $0, $1)"> $input1, $after_onehot),
  [(WhereValue2 $addconst_attr),  (WhereValue3 $floorconst_attr),
   (WhereValue0 $mulconst_attr),  (FpSplatValueOne $floorconst1_attr),
   (WhereValue4 $mulconst1_attr), (WhereValue1 $depth_attr),
   (FpSplatValueOne $onvalue_attr), (FpSplatValueZero $offvalue_attr),
   (IntSplatValueNegOne $shape_attr), (IntSplatValueZero $gatheraxis_attr),
   (AxisAttrNegOne $onehot_axis), (AxisAttrZero $gather_batch_dims),
   (IntegerEleType $before_cast), (FloatEleType $after_cast),
   (FloatEleType $before_cast1), (IntegerEleType $after_cast1),
   (FloatEleType $before_cast2), (FloatEleType $after_cast2),
   (TwoRank $after_cast1), (OneRank $after_squeeze),
   (TwoRank $after_onehot), (OneRank $before_cast2),
   (TwoRank $after_where), (OneRank $after_squeeze1),
   (SameEleType $input1, $after_onehot)]>;

#endif // FUSE_TF_OPS_PATTERN
