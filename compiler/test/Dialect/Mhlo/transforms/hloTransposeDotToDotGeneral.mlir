// RUN: byteir-opt %s -hlo-transpose-dot-to-dot-general | FileCheck %s

func.func @lhs_transpose_dot(%arg0 : tensor<64x128xf32>, %arg1 : tensor<64x32xf32>) -> tensor<128x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %1 = "mhlo.dot"(%0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
    return %1 : tensor<128x32xf32>
}
// CHECK-LABEL: func.func @lhs_transpose_dot
// CHECK-NEXT:  mhlo.dot_general{{.*}}{dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>}
// CHECK:  return

func.func @rhs_transpose_dot(%arg0 : tensor<128x64xf32>, %arg1 : tensor<32x64xf32>) -> tensor<128x32xf32> {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %1 = "mhlo.dot"(%arg0, %0) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
    return %1 : tensor<128x32xf32>
}
// CHECK-LABEL: func.func @rhs_transpose_dot
// CHECK-NEXT:  mhlo.dot_general{{.*}}{dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>}
// CHECK:  return

func.func @lhs_rhs_transpose_dot(%arg0 : tensor<64x128xf32>, %arg1 : tensor<32x64xf32>) -> tensor<128x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %1 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<32x64xf32>) -> tensor<64x32xf32>
    %2 = "mhlo.dot"(%0, %1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
    return %2 : tensor<128x32xf32>
}
// CHECK-LABEL: func.func @lhs_rhs_transpose_dot
// CHECK-NEXT:  mhlo.dot_general{{.*}}{dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>}
// CHECK:  return
