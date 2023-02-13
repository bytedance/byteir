// RUN: byteir-opt %s --canonicalize | FileCheck %s

func.func @add_zero_left(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
  %1 = mhlo.add %c0, %arg0: tensor<128xf32>
  return %1: tensor<128xf32>
}
// CHECK-LABEL: add_zero_left
// CHECK-NEXT: return %arg0

func.func @add_zero_right(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
  %1 = mhlo.add %arg0, %c0: tensor<128xf32>
  return %1: tensor<128xf32>
}
// CHECK-LABEL: add_zero_right
// CHECK-NEXT: return %arg0

func.func @mul_one_left(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
  %1 = mhlo.multiply %c1, %arg0: tensor<128xf32>
  return %1: tensor<128xf32>
}
// CHECK-LABEL: mul_one_left
// CHECK-NEXT: return %arg0

func.func @mul_one_right(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %c1 = mhlo.constant dense<1.000000e+00> : tensor<128xf32>
  %1 = mhlo.multiply %arg0, %c1: tensor<128xf32>
  return %1: tensor<128xf32>
}
// CHECK-LABEL: mul_one_right
// CHECK-NEXT: return %arg0
