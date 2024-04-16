// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true" | FileCheck %s

func.func @cumsum_to_iota_case0() -> tensor<1x16xi64> {
  %0 = mhlo.constant dense<1> : tensor<1x16xi64>
  %1 = mhlo.constant dense<0> : tensor<i64>
  %2 = "mhlo.reduce_window"(%0, %1) ({
    ^bb0(%arg390: tensor<i64>, %arg391: tensor<i64>):
      %2603 = mhlo.add %arg390, %arg391 : tensor<i64>
      mhlo.return %2603 : tensor<i64>
  }) {padding = dense<[[0, 0], [15, 0]]> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[1, 16]> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16xi64>, tensor<i64>) -> tensor<1x16xi64>
  return %2 : tensor<1x16xi64>
}
// CHECK-LABEL: func.func @cumsum_to_iota_case0
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.iota
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.add

func.func @cumsum_to_iota_case1() -> tensor<10x16xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<10x16xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce_window"(%0, %1) ({
    ^bb0(%arg367: tensor<f32>, %arg368: tensor<f32>):
      %3101 = mhlo.add %arg367, %arg368 : tensor<f32>
      mhlo.return %3101 : tensor<f32>
  }) {padding = dense<[[0, 0], [15, 0]]> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[1, 16]> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<10x16xf32>, tensor<f32>) -> tensor<10x16xf32>
  return %2 : tensor<10x16xf32>
}
// CHECK-LABEL: func.func @cumsum_to_iota_case1
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.iota
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.add

func.func @simplify_reduce_to_reshape(%arg0: tensor<1x8xf32>) -> tensor<8xf32> {
  %cst = mhlo.constant dense<0.000> : tensor<f32>
  %0 = "mhlo.reduce"(%arg0, %cst) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1x8xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @simplify_reduce_to_reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return
