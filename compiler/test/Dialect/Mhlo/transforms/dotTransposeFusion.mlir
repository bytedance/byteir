// RUN: byteir-opt %s --mhlo-legalize-dot-to-dot-general -fuse-transpose-into-dot-general --split-input-file | FileCheck %s

func.func @dot_transpose(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
}

// CHECK-LABEL: func.func @dot_transpose
// CHECK-NEXT: %[[V0:.*]] = "mhlo.dot_general"(%arg1, %arg0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
// CHECK-NEXT:  return %[[V0]]

// -----

func.func @dot_general_transpose(%arg0: tensor<128x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x128xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x128xf32>, tensor<128x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
}

// CHECK-LABEL: func.func @dot_general_transpose
// CHECK-NEXT: %[[V0:.*]] = "mhlo.dot_general"(%arg1, %arg0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x30522xf32>, tensor<128x128xf32>) -> tensor<30522x128xf32>
// CHECK-NEXT:  return %[[V0]]

// -----

func.func @dot_general_transpose_1(%arg0: tensor<64x128xf32>, %arg1: tensor<128x30522xf32>) -> tensor<30522x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<64x128xf32>, tensor<128x30522xf32>) -> tensor<64x30522xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<64x30522xf32>) -> tensor<30522x64xf32>
    return %1 : tensor<30522x64xf32>
}

// CHECK-LABEL: func.func @dot_general_transpose_1
// CHECK-NEXT: %[[V0:.*]] = "mhlo.dot_general"(%arg1, %arg0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<128x30522xf32>, tensor<64x128xf32>) -> tensor<30522x64xf32>
// CHECK-NEXT:  return %[[V0]]

// -----

func.func @transpose_transpose_dot_general(%arg0: tensor<2048x2xf32>, %arg1: tensor<1001x2048xf32>) -> tensor<1001x2xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64> } : (tensor<2048x2xf32>) -> tensor<2x2048xf32>
  %1 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64> } : (tensor<1001x2048xf32>) -> tensor<2048x1001xf32>
  %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x2048xf32>, tensor<2048x1001xf32>) -> tensor<2x1001xf32>
  %3 = "mhlo.transpose"(%2) {permutation = dense<[1, 0]> : tensor<2xi64> } : (tensor<2x1001xf32>) -> tensor<1001x2xf32>
  return %3 : tensor<1001x2xf32>
}

// CHECK-LABEL: func.func @transpose_transpose_dot_general
// CHECK-NEXT: %[[V0:.*]] = "mhlo.dot_general"(%arg1, %arg0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1001x2048xf32>, tensor<2048x2xf32>) -> tensor<1001x2xf32>
// CHECK-NEXT:  return %[[V0]]


// -----

func.func @bmm_rrr(%arg0: tensor<384x256x128xf32>, %arg1: tensor<384x128x64xf32>) -> (tensor<384x128x256xf32>, tensor<384x256x64xf32>) {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x128xf32>) -> tensor<384x128x256xf32>
    %1 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 2, 0]> : tensor<3xi64>} : (tensor<384x128x64xf32>) -> tensor<128x64x384xf32>
    %2 = "mhlo.dot_general"(%0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [2], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<384x128x256xf32>, tensor<128x64x384xf32>) -> tensor<384x256x64xf32>
    return %0, %2 : tensor<384x128x256xf32>, tensor<384x256x64xf32>
}

//CHECK-LABEL: func.func @bmm_rrr
// CHECK-NEXT: %[[V0:.*]] = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x128xf32>) -> tensor<384x128x256xf32>
// CHECK-NEXT: %[[V1:.*]] = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x128xf32>, tensor<384x128x64xf32>) -> tensor<384x256x64xf32>
// CHECK-NEXT: return %[[V0]], %[[V1]]
