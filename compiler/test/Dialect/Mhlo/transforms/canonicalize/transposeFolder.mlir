// RUN: byteir-opt %s -canonicalize | FileCheck %s


func.func @transpose_const() -> tensor<5x64x31x95xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @transpose_const
// CHECK-NEXT: mhlo.constant
// CHECK-NOT: mhlo.transpose
// CHECK-NEXT:  return

func.func @broadcast_transpose(%arg0 : tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @broadcast_transpose
// CHECK-NEXT:  mhlo.broadcast_in_dim{{.*}}{broadcast_dimensions = dense<1> : tensor<1xi64>}{{.*}}
// CHECK:  return

func.func @broadcast_transpose_non_dim(%arg0 : tensor<f32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @broadcast_transpose_non_dim
// CHECK-NEXT:  mhlo.broadcast_in_dim{{.*}}{broadcast_dimensions = dense<> : tensor<0xi64>}{{.*}}
// CHECK:  return

func.func @broadcast_transpose_multi_dim(%arg0 : tensor<95x64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<95x64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @broadcast_transpose_multi_dim
// CHECK-NEXT:  mhlo.broadcast_in_dim{{.*}}{broadcast_dimensions = dense<[3, 1]> : tensor<2xi64>}{{.*}}
// CHECK:  return

func.func @transpose_transpose(%arg0 : tensor<31x20x32xf32>) -> tensor<20x32x31xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<20x32x31xf32>
    return %1 : tensor<20x32x31xf32>
}
// CHECK-LABEL: func.func @transpose_transpose
// CHECK-NEXT:  mhlo.transpose{{.*}}{permutation = dense<[1, 2, 0]> : tensor<3xi64>}
// CHECK:  return

func.func @transpose_transpose_to_noop(%arg0 : tensor<31x20x32xf32>) -> tensor<31x20x32xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<31x20x32xf32>) -> tensor<20x31x32xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} : (tensor<20x31x32xf32>) -> tensor<31x20x32xf32>
    return %1 : tensor<31x20x32xf32>
}
// CHECK-LABEL: func.func @transpose_transpose_to_noop
// CHECK:  return %arg0 : tensor<31x20x32xf32>
