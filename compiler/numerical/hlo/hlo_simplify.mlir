// RUN: byteir-opt %s -hlo-simplify --canonicalize --cse -o %t
// RUN: FileCheck %s < %t
// RUN: python3 %S/numerical_test.py %s %t --decimal simplify_dot_general_bmm_case0:5,simplify_dot_general_bmm_case1:5

func.func @simplify_dot_general$gemm$case0(%arg0: tensor<1x1xf32>, %arg1: tensor<1x30xf32>) -> (tensor<1x30xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x1xf32>, tensor<1x30xf32>) -> tensor<1x30xf32>
  return %0 : tensor<1x30xf32>
}
// CHECK-LABEL: @simplify_dot_general$gemm$case0
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_general_bmm_case0(%arg0: tensor<128x1x4xf32>, %arg1: tensor<128x4x50xf32>, %arg2: tensor<128x1x50xf32>) -> tensor<128x1x50xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<128x4x50xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<128x4x50xf32>
  %2 = "mhlo.dot_general"(%arg0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x1x4xf32>, tensor<128x4x50xf32>) -> tensor<128x1x50xf32>
  return %2 : tensor<128x1x50xf32>
}
// CHECK-LABEL: @simplify_dot_general_bmm_case0
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: mhlo.reduce
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @simplify_dot_general_bmm_case1(%arg0: tensor<128x50x4xf32>, %arg1: tensor<128x4x1xf32>, %arg2: tensor<128x50x1xf32>) -> tensor<128x50x1xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<128x4x1xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<128x4x1xf32>
  %2 = "mhlo.dot_general"(%arg0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x50x4xf32>, tensor<128x4x1xf32>) -> tensor<128x50x1xf32>
  return %2 : tensor<128x50x1xf32>
}
// CHECK-LABEL: @simplify_dot_general_bmm_case1
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: mhlo.reduce
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @simplify_dot_general_bmm_case2(%arg0: tensor<128x1x1xf32>, %arg1: tensor<128x1x30xf32>) -> (tensor<128x1x30xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x1x1xf32>, tensor<128x1x30xf32>) -> tensor<128x1x30xf32>
  return %0 : tensor<128x1x30xf32>
}
// CHECK-LABEL: @simplify_dot_general_bmm_case2
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_general_bmm_case3(%arg0: tensor<128x20x1xf32>, %arg1: tensor<128x1x30xf32>) -> (tensor<128x20x30xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x20x1xf32>, tensor<128x1x30xf32>) -> tensor<128x20x30xf32>
  return %0 : tensor<128x20x30xf32>
}
// CHECK-LABEL: @simplify_dot_general_bmm_case3
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_mm_k1_case0(%arg0: tensor<128x1xf32>, %arg1: tensor<1x50xf32>, %arg2: tensor<128x50xf32>) -> tensor<128x50xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<1x50xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<1x50xf32>
  %2 = "mhlo.dot_general"(%arg0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x1xf32>, tensor<1x50xf32>) -> tensor<128x50xf32>
  return %2 : tensor<128x50xf32>
}
// CHECK-LABEL: @simplify_dot_mm_k1_case0
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_mm_k1_case1(%arg0: tensor<128x1xf32>, %arg1: tensor<1x50xf32>, %arg2: tensor<128x50xf32>) -> tensor<128x50xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<1x50xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<1x50xf32>
  %2 = "mhlo.dot"(%arg0, %1) : (tensor<128x1xf32>, tensor<1x50xf32>) -> tensor<128x50xf32>
  return %2 : tensor<128x50xf32>
}
// CHECK-LABEL: @simplify_dot_mm_k1_case1
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return
