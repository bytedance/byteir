// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true" | FileCheck %s

func.func @canonicalize_const_broadcast() -> tensor<1x10x2xi64> {
  %0 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x2xi64>) -> (tensor<1x10x2xi64>)
  return %1 : tensor<1x10x2xi64>
}
// CHECK-LABEL: canonicalize_const_broadcast
// CHECK: %[[V0:.*]] = mhlo.constant dense<[2, 3]> : tensor<2xi64>
// CHECK: "mhlo.broadcast_in_dim"(%[[V0]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1x10x2xi64>

func.func @broacast_reshape_case0(%arg0: tensor<75xi64>) -> tensor<75x1x75xi64> {
  %0 = mhlo.reshape %arg0 : (tensor<75xi64>) -> tensor<1x1x75xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x75xi64>) -> tensor<75x1x75xi64>
  return %1 : tensor<75x1x75xi64>
}
// CHECK-LABEL: @broacast_reshape_case0
// CHECK-NOT: mhlo.reshape
// CHECK: mhlo.broadcast_in_dim
// CHECK-SAME: broadcast_dimensions = dense<2> : tensor<1xi64>
// CHECK-SAME: (tensor<75xi64>) -> tensor<75x1x75xi64>

func.func @broacast_reshape_case1(%arg0: tensor<1x32xf32>) -> tensor<1x32x256xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x32xf32>) -> tensor<1x32x1xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x32x1xf32>) -> tensor<1x32x256xf32>
  return %1 : tensor<1x32x256xf32>
}
// CHECK-LABEL: @broacast_reshape_case1
// CHECK-NOT: mhlo.reshape
// CHECK: mhlo.broadcast_in_dim
// CHECK-SAME: broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
// CHECK-SAME: (tensor<1x32xf32>) -> tensor<1x32x256xf32>

func.func @broadcast_reshape_case2(%arg0: tensor<16x75xf16>) -> tensor<1x16x75x1x1x75xf16> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<16x75xf16>) -> tensor<1x16x1x1x1x75xf16>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi64>} : (tensor<1x16x1x1x1x75xf16>) -> tensor<1x16x75x1x1x75xf16>
  return %1 : tensor<1x16x75x1x1x75xf16>
}
// CHECK-LABEL: @broadcast_reshape_case2
// CHECK-NOT: mhlo.reshape
// CHECK: mhlo.broadcast_in_dim
// CHECK-SAME: broadcast_dimensions = dense<[1, 5]> : tensor<2xi64>
// CHECK-SAME: (tensor<16x75xf16>) -> tensor<1x16x75x1x1x75xf16>

func.func @broadcsat_reshape_case3(%arg0: tensor<1x1x32xf32>) -> tensor<1x76x1x32xf32> {
  %0 = mhlo.reshape %arg0 : (tensor<1x1x32xf32>) -> tensor<1x1x1x32xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x32xf32>) -> tensor<1x76x1x32xf32>
  return %1 : tensor<1x76x1x32xf32>
}
// CHECK-LABEL: @broadcsat_reshape_case3
// CHECK-NOT: mhlo.reshape
// CHECK: mhlo.broadcast_in_dim
// CHECK-SAME: broadcast_dimensions = dense<[0, 1, 3]> : tensor<3xi64>
// CHECK-SAME: (tensor<1x1x32xf32>) -> tensor<1x76x1x32xf32>

func.func @broadcast_reshape_not_fold_case0(%arg0: tensor<1x32x1xf32>) -> tensor<1x32x256xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x32xf32>) -> tensor<1x32x256xf32>
  return %1 : tensor<1x32x256xf32>
}
// CHECK-LABEL: @broadcast_reshape_not_fold_case0
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
