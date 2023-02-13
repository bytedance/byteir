// RUN: byteir-opt %s -test-broadcast-dense-elements-attr | FileCheck %s

func.func @case0() -> tensor<4x2xi64> {
  %0 = mhlo.constant dense<[2, 3]> : tensor<2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> (tensor<4x2xi64>)
  return %1 : tensor<4x2xi64>
}
// CHECK-LABEL: @case0
// CHECK{LITERAL}: [[2, 3], [2, 3], [2, 3], [2, 3]]
// CHECK-NOT: mhlo.broadcast_in_dim

func.func @case1() -> tensor<1x10x2xi64> {
  %0 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x2xi64>) -> (tensor<1x10x2xi64>)
  return %1 : tensor<1x10x2xi64>
}
// CHECK-LABEL: @case1
// CHECK{LITERAL}: [[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]]
// CHECK-NOT: mhlo.broadcast_in_dim

func.func @case2() -> tensor<5x1x2xi64> {
  %0 = mhlo.constant dense<[[[2, 3]]]> : tensor<1x1x2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x2xi64>) -> (tensor<5x1x2xi64>)
  return %1 : tensor<5x1x2xi64>
}
// CHECK-LABEL: @case2
// CHECK{LITERAL}: [[[2, 3]], [[2, 3]], [[2, 3]], [[2, 3]], [[2, 3]]]
// CHECK-NOT: mhlo.broadcast_in_dim

func.func @case3() -> tensor<2x1x5xi64> {
  %0 = mhlo.constant dense<[[[2, 3]]]> : tensor<1x1x2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1, 2, 0]> : tensor<3xi64>} : (tensor<1x1x2xi64>) -> (tensor<2x1x5xi64>)
  return %1 : tensor<2x1x5xi64>
}
// CHECK-LABEL: @case3
// CHECK{LITERAL}: [[[2, 2, 2, 2, 2]], [[3, 3, 3, 3, 3]]]
// CHECK-NOT: mhlo.broadcast_in_dim

func.func @case4() -> tensor<5x1x2xi64> {
  %0 = mhlo.constant dense<2> : tensor<i64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i64>) -> (tensor<5x1x2xi64>)
  return %1 : tensor<5x1x2xi64>
}
// CHECK-LABEL: @case4
// CHECK: mhlo.constant dense<2> : tensor<5x1x2xi64>
// CHECK-NOT: mhlo.broadcast_in_dim

func.func @case5() -> tensor<5x1x2xi64> {
  %0 = mhlo.constant dense<2> : tensor<1x2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xi64>) -> (tensor<5x1x2xi64>)
  return %1 : tensor<5x1x2xi64>
}
// CHECK-LABEL: @case5
// CHECK: mhlo.constant dense<2> : tensor<5x1x2xi64>
// CHECK-NOT: mhlo.broadcast_in_dim
