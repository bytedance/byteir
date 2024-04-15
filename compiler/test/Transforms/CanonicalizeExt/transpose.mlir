// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true fold-limit=-1" | FileCheck %s

func.func @eliminate_splat_constant_transpose() -> tensor<2x1x4x3xi32> {
  %0 = mhlo.constant dense<0> : tensor<1x2x3x4xi32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %1: tensor<2x1x4x3xi32>
}
// CHECK-LABEL: eliminate_splat_constant_transpose
// CHECK-NEXT: %0 = mhlo.constant dense<0> : tensor<2x1x4x3xi32>

func.func @transpose_non_splat_constant_2d() -> tensor<2x1xf32> {
  %0 = mhlo.constant dense<[[1.0000, 0.0000]]> : tensor<1x2xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
// CHECK-LABEL: transpose_non_splat_constant_2d
// CHECK{LITERAL}:  mhlo.constant dense<[[1.000000e+00], [0.000000e+00]]> : tensor<2x1xf32>

func.func @transpose_non_splat_constant_3d() -> tensor<2x2x2xf32> {
  %0 = mhlo.constant dense<[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]> : tensor<2x2x2xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[2, 1, 0]> : tensor<3xi64>} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  return %1 : tensor<2x2x2xf32>
}
// CHECK-LABEL: transpose_non_splat_constant_3d
// CHECK{LITERAL}:  mhlo.constant dense<[[[0.000000e+00, 4.000000e+00], [2.000000e+00, 6.000000e+00]], [[1.000000e+00, 5.000000e+00], [3.000000e+00, 7.000000e+00]]]> : tensor<2x2x2xf32>


func.func @transpose_reshape_transpose(%arg0: tensor<2x32x128x256xf16>) -> (tensor<64x256x128xf16>, tensor<64x128x256xf16>) {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<2x32x128x256xf16>) -> tensor<2x32x256x128xf16>
  %1 = mhlo.reshape %0 : (tensor<2x32x256x128xf16>) -> tensor<64x256x128xf16>
  %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<64x256x128xf16>) -> tensor<64x128x256xf16>
  return %1, %2 : tensor<64x256x128xf16>, tensor<64x128x256xf16>
}
// CHECK-LABEL: func.func @transpose_reshape_transpose
// CHECK-NEXT: mhlo.transpose
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return
