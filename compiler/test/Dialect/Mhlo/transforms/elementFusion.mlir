// RUN: byteir-opt %s -fuse-element | FileCheck %s
// RUN: byteir-opt %s -fuse-element="cluster-single-op" | FileCheck %s --check-prefix CHECK-SINGLE

func.func @mhlo_element(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.abs"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.add"(%arg0, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%1, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_element
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.abs
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_elementwise_fusion__}
// CHECK:  return

func.func @mhlo_element_broadcast(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<3x4xf32>
  %2 = "mhlo.abs"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%1, %arg2) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  %4 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<3x4xf32>
  %5 = "mhlo.add"(%3, %4) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %5 : tensor<3x4xf32>
}
// CHECK-LABEL: func.func @mhlo_element_broadcast
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.abs
// CHECK-NEXT:    mhlo.return
// CHECK:  mhlo.fusion
// CHECK-NEXT:    mhlo.broadcast_in_dim

// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.broadcast_in_dim
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_elementwise_fusion__}
// CHECK:  return


func.func @mhlo_element_reshape(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<2x2xf32>) -> tensor<4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.abs"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.reshape"(%arg2) : (tensor<2x2xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%0, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %3 : tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_element_reshape
// CHECK:       mhlo.fusion
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.abs
// CHECK-NEXT:    mhlo.reshape
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_elementwise_fusion__}
// CHECK:  return

func.func @shared_constant(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>, %arg3 : tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<4xf32>
  %1 = "mhlo.add"(%arg0, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.abs"(%1) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "mhlo.add"(%1, %3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %5 = "mhlo.dot"(%arg3, %0) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %4, %5 : tensor<4xf32>, tensor<4xf32>
}
// CHECK-LABEL: func.func @shared_constant
// CHECK:       mhlo.fusion
// CHECK-NEXT:    mhlo.constant
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.abs
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_elementwise_fusion__}
// CHECK:  return

// test not crash
func.func private @empty(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>, %arg2 : tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK-LABEL: func.func private @empty

func.func @mhlo_single_op(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_single_op
//   CHECK-NEXT: mhlo.add
//   CHECK-NEXT: return
// CHECK-SINGLE-LABEL: func.func @mhlo_single_op
//   CHECK-SINGLE: mhlo.fusion
//     CHECK-SINGLE-NEXT: mhlo.add
//     CHECK-SINGLE-NEXT: mhlo.return
//   CHECK-SINGLE: {__byteir_elementwise_fusion__}
//   CHECK-SINGLE: return

func.func @mhlo_single_broadcast(%arg0 : tensor<1x128xi64>) -> tensor<2x128xi64> {
  %1 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128xi64>) -> tensor<2x128xi64>
  return %1 : tensor<2x128xi64>
}
// CHECK-LABEL: func.func @mhlo_single_broadcast
//   CHECK-NEXT: mhlo.broadcast_in_dim
//   CHECK-NEXT: return
// CHECK-SINGLE-LABEL: func.func @mhlo_single_broadcast
//   CHECK-SINGLE: mhlo.fusion
//     CHECK-SINGLE-NEXT: mhlo.broadcast_in_dim
//     CHECK-SINGLE-NEXT: mhlo.return
//   CHECK-SINGLE: {__byteir_elementwise_fusion__}
//   CHECK-SINGLE: return