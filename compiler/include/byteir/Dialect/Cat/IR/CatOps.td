//===-- CatOps.td - Cat dialect operation definitions --*- tablegen -*---===//
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
//
// Defines operations of the Cat dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_CAT_IR_CAT_OPS
#define BYTEIR_DIALECT_CAT_IR_CAT_OPS

include "byteir/Dialect/Cat/IR/CatBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Cat Dialect operations.
//===----------------------------------------------------------------------===//

// Base class
class Cat_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cat_Dialect, mnemonic, traits>;

def Cat_CastOp : Cat_Op<"cast", [Cat_CatOpInterface, Pure]> {
  let summary = "type cast operator";
  let arguments = (ins AnyTensor : $input);
  let results = (outs AnyTensor : $output);
}

def Cat_Conv2dOp : Cat_Op<"conv2d", [Cat_CatOpInterface, Pure]> {
  let summary = "conv2d operator";
  let arguments = (ins AnyTensor : $input,
                       AnyTensor : $weight,
                       StrAttr : $layout, 
                       I64ElementsAttr : $stride, 
                       I64ElementsAttr : $padding, 
                       OptionalAttr<I64ElementsAttr> : $lhs_dilation, 
                       OptionalAttr<I64ElementsAttr> : $rhs_dilation);
  let results = (outs AnyTensor : $output);
}

def Cat_Conv2dBiasOp : Cat_Op<"conv2d_bias", [Cat_CatOpInterface, Pure]> {
  let summary = "conv2d+bias operator";
  let arguments = (ins AnyTensor : $input,
                       AnyTensor : $weight,
                       AnyTensor : $bias,
                       StrAttr : $layout,
                       I64ElementsAttr : $stride,
                       I64ElementsAttr : $padding,
                       OptionalAttr<I64ElementsAttr> : $lhs_dilation,
                       OptionalAttr<I64ElementsAttr> : $rhs_dilation);
  let results = (outs AnyTensor : $output);
}

def Cat_Conv2dBiasReluOp : Cat_Op<"conv2d_bias_relu", [Cat_CatOpInterface, Pure]> {
  let summary = "conv2d+bias+relu operator";
  let arguments = (ins AnyTensor : $input,
                       AnyTensor : $weight,
                       AnyTensor : $bias,
                       StrAttr : $layout,
                       I64ElementsAttr : $stride,
                       I64ElementsAttr : $padding,
                       OptionalAttr<I64ElementsAttr> : $lhs_dilation,
                       OptionalAttr<I64ElementsAttr> : $rhs_dilation);
  let results = (outs AnyTensor : $output);
}

def Cat_Conv2dBiasAddReluOp : Cat_Op<"conv2d_bias_add_relu", [Cat_CatOpInterface, Pure]> {
  let summary = "conv2d+bias+add+relu operator";
  let arguments = (ins AnyTensor : $input,
                       AnyTensor : $weight,
                       AnyTensor : $bias,
                       AnyTensor : $res,
                       StrAttr : $layout,
                       I64ElementsAttr : $stride,
                       I64ElementsAttr : $padding,
                       OptionalAttr<I64ElementsAttr> : $lhs_dilation,
                       OptionalAttr<I64ElementsAttr> : $rhs_dilation);
  let results = (outs AnyTensor : $output);
}

def Cat_NchwToNhwcOp : Cat_Op<"nchw2nhwc", [Cat_CatOpInterface, Pure]> {
  let summary = "nchw-to-nhwc relayout operator";
  let arguments = (ins AnyTensor : $input);
  let results = (outs AnyTensor : $output);
}

// TODO: add verifier for each op
def Cat_BMMRRRAddOp : Cat_Op<"bmm_rrr_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (rrr)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRRCAddOp : Cat_Op<"bmm_rrc_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (rrc)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRCRAddOp : Cat_Op<"bmm_rcr_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (rcr)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRCCAddOp : Cat_Op<"bmm_rcc_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (rcc)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCRRAddOp : Cat_Op<"bmm_crr_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (crr)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCRCAddOp : Cat_Op<"bmm_crc_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (crc)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCCRAddOp : Cat_Op<"bmm_ccr_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (ccr)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCCCAddOp : Cat_Op<"bmm_ccc_add", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + add operator (ccc)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $add);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRRRPermuteOp : Cat_Op<"bmm_rrr_permute", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + permute0213 operator";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       I64Attr : $shape);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_BMMRCRPermuteOp : Cat_Op<"bmm_rcr_permute", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul + permute0213 operator";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       I64Attr : $shape);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_BMMSoftmaxBMMOp : Cat_Op<"bmm_softmax_bmm", [Cat_CatOpInterface, Pure]> {
  let summary = "flash attention operator";
  let arguments = (ins AnyTensor : $inputQ,
                       AnyTensor : $inputK,
                       AnyTensor : $inputV);
  let results = (outs AnyTensor : $output);
}

def Cat_BatchNormOp : Cat_Op<"batch_norm", [Cat_CatOpInterface, Pure]> {
  let summary = "batch norm operator";
  let arguments = (ins AnyTensor : $input,
                       AnyTensor : $gamma,
                       AnyTensor : $beta,
                       AnyTensor : $mean, 
                       AnyTensor : $var);
  let results = (outs AnyTensor : $output);
}

def Cat_Pooling2dOp : Cat_Op<"pooling2d", [Cat_CatOpInterface, Pure]> {
  let summary = "pooling2d operator";
  let arguments = (ins AnyTensor : $input,
                       I64Attr : $window_stride,
                       I64Attr : $padding,
                       I64Attr : $kernel_size,
                       StrAttr : $reduce_func);
  let results = (outs AnyTensor : $output);
}

def Cat_ReluOp : Cat_Op<"relu", [Cat_CatOpInterface, Pure]> {
  let summary = "relu operator";
  let arguments = (ins AnyTensor : $input);
  let results = (outs AnyTensor : $output);
}

def Cat_ReduceOp : Cat_Op<"reduce", [Cat_CatOpInterface, Pure]> {
  let summary = "reduce operator";
  let arguments = (ins AnyTensor : $input, I64ElementsAttr : $dims, StrAttr : $reduce_type);
  let results = (outs AnyTensor : $output);
}

def Cat_SliceOp : Cat_Op<"slice", [Cat_CatOpInterface, Pure]> {
  let summary = "slice operator";
  let arguments = (ins AnyTensor : $input, I64ElementsAttr : $start_indices, I64ElementsAttr : $end_indices);
  let results = (outs AnyTensor : $output);
}

def Cat_GemmRRRBiasOp : Cat_Op<"gemm_rrr_bias", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm_rrr_bias operator";
  let description =
      "bias must be a length (n) vector, given that lhsxrhs is (m, n)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $bias);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_GemmRCRBiasOp : Cat_Op<"gemm_rcr_bias", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm_rcr_bias operator";
  let description =
      "bias must be a length (n) vector, given that lhsxrhs is (m, n)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $bias);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_GemmRRROp : Cat_Op<"gemm_rrr", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm_rrr operator";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_GemmRCROp : Cat_Op<"gemm_rcr", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm_rcr operator";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_GemmRCRPermuteOp : Cat_Op<"gemm_rcr_permute", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm_rcr + permute0213 operator, output layout is [m / t1, t1, t2, n / t2]";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       I64Attr : $t1,
                       I64Attr : $t2);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_GemmRRRPermuteOp : Cat_Op<"gemm_rrr_permute", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm_rrr + permute0213 operator, output layout is [m / t1, t1, t2, n / t2]";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       I64Attr : $t1,
                       I64Attr : $t2);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_LayerNormOp : Cat_Op<"layernorm", [Cat_CatOpInterface, Pure]> {
  let summary = "layernorm operator";
  let arguments = (ins AnyTensor : $input,
                       AnyTensor : $beta,
                       AnyTensor : $gamma,
                       I64ArrayAttr:$axis,
                       F64Attr:$epsilon);
  let results = (outs AnyTensor : $output);
}

def Cat_SoftmaxOp : Cat_Op<"softmax", [Cat_CatOpInterface, Pure]> {
  let summary = "softmax operator";
  let arguments = (ins AnyTensor : $input, I64Attr : $dim);
  let results = (outs AnyTensor : $output);
}

def Cat_UnaryElementwiseOp : Cat_Op<"unary_elementwise", [Cat_CatOpInterface, Pure]> {
  let summary = "unary elementwise operator";
  let arguments = (ins AnyTensor : $input, StrAttr : $op_type);
  let results = (outs AnyTensor : $output);
}

def Cat_BinaryElementwiseOp : Cat_Op<"binary_elementwise", [Cat_CatOpInterface, Pure]> {
  let summary = "binary elementwise operator";
  let arguments = (ins AnyTensor : $lhs, AnyTensor : $rhs, StrAttr : $op_type);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRRROp : Cat_Op<"bmm_rrr", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (rrr layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRRCOp : Cat_Op<"bmm_rrc", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (rrc layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRCROp : Cat_Op<"bmm_rcr", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (rcr layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMRCCOp : Cat_Op<"bmm_rcc", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (rcc layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCRROp : Cat_Op<"bmm_crr", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (crr layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCRCOp : Cat_Op<"bmm_crc", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (crc layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCCROp : Cat_Op<"bmm_ccr", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (ccr layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_BMMCCCOp : Cat_Op<"bmm_ccc", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator (ccc layout)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $output);
}

def Cat_ConcatenateOp : Cat_Op<"concatenate", [Cat_CatOpInterface, Pure]> {
  let summary = "concatenate operator";
  let arguments = (ins Variadic<AnyTensor> : $val,
                       I64Attr:$dimension);
  let results = (outs AnyTensor : $output);
}

def Cat_WhereOp : Cat_Op<"where", [Cat_CatOpInterface, Pure]> {
  let summary = "where operator";
  let arguments = (ins AnyTensor : $condition,
                       AnyTensor : $on_true,
                       AnyTensor : $on_false);
  let results = (outs AnyTensor : $output);
}

#endif // BYTEIR_DIALECT_CAT_IR_CAT_OPS
