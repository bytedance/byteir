#ifndef FUSE_ON_TORCH_PATTERN
#define FUSE_ON_TORCH_PATTERN

include "mlir/IR/PatternBase.td"
include "torch-mlir/Dialect/Torch/IR/TorchOps.td"

def OneIntegerAttr : Constraint<CPred<"$0.getValue() == 1">, "Integer 1">;
def ThreeIntegerAttr : Constraint<CPred<"$0.getValue() == 3">, "Integer 3">;
def TrueAttr : Constraint<CPred<"$0.getValue() == 1">, "True">;
def FalseAttr : Constraint<CPred<"$0.getValue() == 0">, "False">;

// ==============================================================================

def createGeluTanh : NativeCodeCall<"createGeluTanh($_builder, $_loc, $0, $1)">;

def TorchGeluTanhPattern : Pat<
    (Torch_AtenMulTensorOp:$output
        (Torch_AtenMulTensorOp
            $input,
            (Torch_NonValueTensorLiteralOp $const_5)
        ),
        (Torch_AtenAddTensorOp
            (Torch_AtenTanhOp
                (Torch_AtenMulTensorOp
                    (Torch_AtenAddTensorOp
                        $input,
                        (Torch_AtenMulTensorOp
                            (Torch_AtenPowTensorScalarOp
                                $input,
                                (Torch_ConstantIntOp $int3)
                            ),
                            (Torch_NonValueTensorLiteralOp $const_4)
                        ),
                        (Torch_ConstantIntOp $int1_1)
                    ),
                    (Torch_NonValueTensorLiteralOp $const_7)
                )
            ),
            (Torch_NonValueTensorLiteralOp $const_1),
            (Torch_ConstantIntOp $int1)
        )
    ),
    (createGeluTanh $output, $input),
    [(OneIntegerAttr $int1), (OneIntegerAttr $int1_1), (ThreeIntegerAttr $int3),
     (Constraint<CPred<"isSplatValue(dyn_cast<DenseIntElementsAttr>($0), 1)">, ""> $const_1),
     (Constraint<CPred<"isSplatValue(dyn_cast<DenseFPElementsAttr>($0), 0.5)">, ""> $const_5),
     (Constraint<CPred<"isSplatCloseToValue(dyn_cast<DenseFPElementsAttr>($0), 4.471500e-02, 0.0001)">, ""> $const_4),
     (Constraint<CPred<"isSplatCloseToValue(dyn_cast<DenseFPElementsAttr>($0), 0.797884583, 0.0001)">, ""> $const_7)]
>;

// ==============================================================================

def createGeluErf : NativeCodeCall<"createGeluErf($_builder, $_loc, $0, $1)">;

def TorchGeluErfPattern : Pat<
    (Torch_AtenMulTensorOp:$output
        (Torch_AtenMulTensorOp
            $input,
            (Torch_NonValueTensorLiteralOp $const_5)
        ),
        (Torch_AtenAddTensorOp
            (Torch_AtenErfOp
                (Torch_AtenDivTensorOp
                    $input,
                    (Torch_NonValueTensorLiteralOp $const_2)
                )
            ),
            (Torch_NonValueTensorLiteralOp $const_1),
            (Torch_ConstantIntOp $int1)
        )
    ),
    (createGeluErf $output, $input),
    [(OneIntegerAttr $int1),
     (Constraint<CPred<"isSplatValue(dyn_cast<DenseFPElementsAttr>($0), 0.5)">, ""> $const_5),
     (Constraint<CPred<"isSplatValue(dyn_cast<DenseFPElementsAttr>($0), 1.0)">, ""> $const_1),
     (Constraint<CPred<"isSplatCloseToValue(dyn_cast<DenseFPElementsAttr>($0), 1.4142135623730951, 0.0001)">, ""> $const_2)]
>;

// ==============================================================================

def createLayerNorm : NativeCodeCall<"createLayerNorm($_builder, $_loc, $0, $1, $2, $3, $4, $5, $6)">;

def TorchLayerNormPattern : Pat<
    (Torch_AtenAddTensorOp:$output
        (Torch_AtenDivTensorOp
            (Torch_AtenMulTensorOp
                $weight,
                (Torch_AtenSubTensorOp
                    $input,
                    (Torch_AtenMeanDimOp
                        $input,
                        $list,
                        (Torch_ConstantBoolOp $true_value),
                        (Torch_ConstantNoneOp)
                    ),
                    (Torch_ConstantIntOp $int1_2)
                )
            ),
            (Torch_AtenAddTensorOp
                (Torch_AtenStdDimOp
                    $input,
                    $list,
                    (Torch_ConstantBoolOp:$false_op $false_value),
                    (Torch_ConstantBoolOp $true_value_1)
                ),
                (Torch_NonValueTensorLiteralOp $epsilon),
                (Torch_ConstantIntOp $int1_1)
            )
        ),
        $bias,
        (Torch_ConstantIntOp $int1)
    ),
    (createLayerNorm $output, $input, $list, $weight, $bias, (NativeCodeCall<"createLayerNormEpsilon($_builder, $_loc, $0)"> $epsilon), $false_op),
    [(TrueAttr $true_value), (TrueAttr $true_value_1), (FalseAttr $false_value),
     (OneIntegerAttr $int1), (OneIntegerAttr $int1_1), (OneIntegerAttr $int1_2)]
>;

#endif // FUSE_ON_TORCH_PATTERN
