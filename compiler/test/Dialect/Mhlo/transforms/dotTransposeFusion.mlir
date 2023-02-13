// RUN: byteir-opt %s -fuse-dot-transpose | FileCheck %s

func.func @dot_transpose(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
}
// CHECK-LABEL: func.func @dot_transpose
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:  mhlo.dot
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.return
// CHECK-NEXT:  }) {__byre__lhs_contracting_dimension = 1 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}
// CHECK:  return

func.func @dot_general_transpose(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
}
// CHECK-LABEL: func.func @dot_general_transpose
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:  mhlo.dot_general
// CHECK-NEXT:  mhlo.transpose
// CHECK-NEXT:  mhlo.return
// CHECK-NEXT:  }) {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"}
// CHECK:  return

