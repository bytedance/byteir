// RUN: byteir-opt %s --linalg-tensor-opt --split-input-file | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<1024xf32>
    %2 = mhlo.add %1, %arg2 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
}

// CHECK-LABEL:  func.func @main(
// CHECK-SAME:      %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>, %[[ARG2:.+]]: tensor<1024xf32>
// CHECK:           linalg.generic
// CHECK-SAME:          ins(%[[ARG0]], %[[ARG1]]
// CHECK:             arith.addf
// CHECK-NEXT:        arith.addf
// CHECK-NEXT:        linalg.yield
// CHECK-NOT:       linalg.generic

// -----

module {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<1xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1024xf32>
    %2 = mhlo.add %1, %arg2 : tensor<1024xf32>
    return %2 : tensor<1024xf32>
  }
}

// CHECK-LABEL:  func.func @main(
// CHECK-SAME:      %[[ARG0:.+]]: tensor<1xf32>, %[[ARG1:.+]]: tensor<1xf32>, %[[ARG2:.+]]: tensor<1024xf32>
// CHECK:           linalg.generic
// CHECK:             arith.addf
// CHECK-NEXT:        arith.addf
// CHECK-NEXT:        linalg.yield
// CHECK-NOT:       linalg.generic

// -----

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1024xf32>) -> tensor<1024xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<f32>
    %1 = mhlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1024xf32>
    %3 = mhlo.add %2, %arg2 : tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
}

// CHECK-LABEL:  func.func @main(
// CHECK-SAME:      %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>, %[[ARG2:.+]]: tensor<1024xf32>
// CHECK:           linalg.generic
// CHECK-SAME:          ins(%[[ARG0]], %[[ARG1]]
// CHECK:             arith.addf
// CHECK-NEXT:        arith.addf
// CHECK-NEXT:        linalg.yield
// CHECK-NOT:       linalg.generic
