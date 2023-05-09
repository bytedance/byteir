// RUN: byteir-opt %s -canonicalize-ext="blind-fold" | FileCheck %s

// CHECK-LABEL: func @slice_constant_3x4_offsets
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[CONST:.+]] = arith.constant dense<{{\[}}[1.200000e+01, 1.300000e+01], [3.000000e+00, 5.000000e+00]]> : tensor<2x2xf32>
//       CHECK:   return %[[CONST]] :  tensor<2x2xf32>
func.func @slice_constant_3x4_offsets(%arg0 : tensor<3x4xf32>) -> tensor<2x2xf32>
{
  %cst = arith.constant dense<[[10.0, 9.0, 8.0, 7.0], [11.0, 12.0, 13.0, 14.0], [1.0, 3.0, 5.0, 7.0]]> : tensor<3x4xf32>
  %slice = tensor.extract_slice %cst[1, 1] [2, 2] [1, 1] : tensor<3x4xf32> to tensor<2x2xf32>
  return %slice : tensor<2x2xf32>
}


