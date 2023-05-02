// RUN: byteir-opt %s -fuse-bmm-dimension | FileCheck %s

func.func @dot_general(%arg0 : tensor<12x4x64x64xf32>, %arg1 : tensor<12x4x64x32xf32>) -> (tensor<12x4x64x32xf32>) {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>} : (tensor<12x4x64x64xf32>, tensor<12x4x64x32xf32>) -> tensor<12x4x64x32xf32>
    return %0 : tensor<12x4x64x32xf32>
}

// CHECK-LABEL: dot_general
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.dot_general
// CHECK-NEXT: mhlo.reshape
