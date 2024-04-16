// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true" | FileCheck %s

func.func @concat_const_folding_with_dup_value() -> tensor<3x2xi64> {
  %0 = mhlo.constant dense<1> : tensor<1x2xi64>
  %1 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
  %3 = "mhlo.concatenate"(%0, %1, %1) {dimension = 0 : i64} : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<3x2xi64>
  return %3 : tensor<3x2xi64>
}
// CHECK-LABEL: concat_const_folding_with_dup_value
// CHECK: %0 = mhlo.constant dense<{{\[}}[1, 1], [2, 3], [2, 3]{{\]}}> : tensor<3x2xi64>
// CHECK: return %0 : tensor<3x2xi64>

func.func @concat_const_folding_case1(%arg0: tensor<1x2xi64>) -> tensor<5x2xi64> {
  %0 = mhlo.constant dense<1> : tensor<1x2xi64>
  %1 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
  %2 = mhlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
  %4 = "mhlo.concatenate"(%0, %1, %arg0, %2, %2) {dimension = 0 : i64} : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<5x2xi64>
  return %4 : tensor<5x2xi64>
}
// CHECK-LABEL: concat_const_folding_case1
// CHECK:  %0 = mhlo.constant dense<{{\[}}[1, 1], [2, 3]{{\]}}> : tensor<2x2xi64>
// CHECK:  %1 = mhlo.constant dense<{{\[}}[3, 4], [3, 4]{{\]}}> : tensor<2x2xi64>

func.func @canonicalize_concat_broadcast(%arg0: tensor<512xf32>) -> tensor<512x1096xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<512x72xf32>
  %1 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<512x1024xf32>
  %2 = "mhlo.concatenate"(%0, %1) {dimension = 1 : i64} : (tensor<512x72xf32>, tensor<512x1024xf32>) -> tensor<512x1096xf32>
  return %2 : tensor<512x1096xf32>
}
// CHECK-LABEL: @canonicalize_concat_broadcast
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-SAME: (tensor<512xf32>) -> tensor<512x1096xf32>
