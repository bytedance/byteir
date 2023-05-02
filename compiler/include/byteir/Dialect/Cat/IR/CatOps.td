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

def Cat_BatchMatmulOp : Cat_Op<"batch_matmul", [Cat_CatOpInterface, Pure]> {
  let summary = "batch matmul operator";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       StrAttr : $layout);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_BMMPermuteOp : Cat_Op<"bmm_permute", [Cat_CatOpInterface, Pure]> {
  let summary = "flash attention operator";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       I64Attr : $shape,
                       StrAttr : $layout);
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
  let arguments = (ins AnyTensor : $input, I64ElementsAttr : $dims);
  let results = (outs AnyTensor : $output);
}

def Cat_GemmBiasOp : Cat_Op<"gemm_bias", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm+bias operator";
  let description =
      "bias must be a length (n) vector, given that lhsxrhs is (m, n)";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       AnyTensor : $bias,
                       StrAttr : $layout);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
}

def Cat_GemmOp : Cat_Op<"gemm", [Cat_CatOpInterface, Pure]> {
  let summary = "gemm operator";
  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       StrAttr : $layout);
  let results = (outs AnyTensor : $output);

  let hasVerifier = 1;
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

#endif // BYTEIR_DIALECT_CAT_IR_CAT_OPS
