// RUN: byteir-opt %s --fuse-concat-slice --split-input-file | FileCheck %s

func.func @concat_and_reshape_mul(%arg0: tensor<4x11xf32>, %arg1: tensor<32x44xf32>, %arg2: tensor<4x5xf32>) -> tensor<32x44xf32> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x2xf32>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 11]> : tensor<2xi64>, start_indices = dense<[0, 7]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x4xf32>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 10]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<2x10xf32>
  %e = mhlo.reshape %2 : (tensor<2x10xf32>) -> tensor<4x5xf32>
  %3 = "mhlo.concatenate"(%e, %0, %1) {dimension = 1 : i64} : (tensor<4x5xf32>, tensor<4x2xf32>, tensor<4x4xf32>) -> tensor<4x11xf32>
  %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<32x4x11xf32>
  %6 = mhlo.reshape %4 : (tensor<32x4x11xf32>) -> tensor<32x44xf32>
  %7 = mhlo.multiply %6, %arg1 : tensor<32x44xf32>
  return %7 : tensor<32x44xf32> 
}

// CHECK-LABEL: func.func @concat_and_reshape_mul
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.slice
// CHECK-NEXT:    mhlo.slice
// CHECK-NEXT:    mhlo.slice
// CHECK-NEXT:    mhlo.reshape
// CHECK-NEXT:    mhlo.concatenate
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_elementwise_fusion__}
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.multiply

// -----

func.func @slice_elementwise(%arg0: tensor<4x11xf32>, %arg1: tensor<4x2xf32>) -> (tensor<2x2xf32>, tensor<4x2xf32>) {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<2x4xf32>
  %1 = mhlo.reshape %0 : (tensor<2x4xf32>) -> tensor<4x2xf32>
  %2 = mhlo.multiply %1, %arg1 : tensor<4x2xf32>
  %3 = "mhlo.slice"(%2) {limit_indices = dense<[4, 2]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x2xf32>) -> tensor<2x2xf32>
  %4 = mhlo.abs %3 : tensor<2x2xf32>
  return %4, %2 : tensor<2x2xf32>, tensor<4x2xf32>
}

// CHECK-LABEL: func.func @slice_elementwise
// CHECK-NOT: mhlo.fusion
// CHECK: return

// -----

func.func @elementwise_concat(%arg0: tensor<4x5xf32>, %arg1: tensor<4x2xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<32x4x11xf32>) -> tensor<32x4x11xf32> {
  %0 = mhlo.abs %arg0 : tensor<4x5xf32>
  %1 = mhlo.abs %arg1 : tensor<4x2xf32>
  %2 = mhlo.abs %arg2 : tensor<4x4xf32>
  %3 = "mhlo.concatenate"(%0, %1, %2) {dimension = 1 : i64} : (tensor<4x5xf32>, tensor<4x2xf32>, tensor<4x4xf32>) -> tensor<4x11xf32>
  %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<32x4x11xf32>
  %5 = mhlo.multiply %4, %arg3 : tensor<32x4x11xf32>
  return %5 : tensor<32x4x11xf32> 
}

// CHECK-LABEL: func.func @elementwise_concat
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.abs
// CHECK-NEXT: mhlo.fusion
// CHECK-NEXT: mhlo.concatenate
// CHECK: {__byteir_elementwise_fusion__}
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply

// -----

func.func @multi_concat(%arg0: tensor<4x5xf32>, %arg1: tensor<4x2xf32>, %arg2: tensor<4x5xf32>) -> tensor<4x7xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<4x5xf32>, tensor<4x2xf32>) -> tensor<4x7xf32>
  %1 = "mhlo.concatenate"(%arg1, %arg2) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<4x5xf32>) -> tensor<4x7xf32>
  %2 = mhlo.multiply %0, %1 : tensor<4x7xf32>
  return %2 : tensor<4x7xf32>
}

// CHECK-LABEL: func.func @multi_concat
// CHECK-NEXT: mhlo.fusion
// CHECK-NEXT:  mhlo.concatenate
// CHECK-NEXT:  mhlo.return
// CHECK:  {__byteir_elementwise_fusion__}
// CHECK-NEXT: mhlo.fusion
// CHECK-NEXT:  mhlo.concatenate
// CHECK-NEXT:  mhlo.return
// CHECK:  {__byteir_elementwise_fusion__}
// CHECK-NEXT: mhlo.multiply

// -----

func.func @slice_reshape_slice_concat(%arg0: tensor<16x32xf32>, %arg1: tensor<8xf32>) -> tensor<4x7xf32> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[8, 16]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<16x32xf32>) -> tensor<8x16xf32>
  %1 = mhlo.reshape %0 : (tensor<8x16xf32>) -> tensor<4x32xf32>
  %2 = "mhlo.slice"(%1) {limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x32xf32>) -> tensor<4x5xf32>
  %3 = mhlo.reshape %arg1 : (tensor<8xf32>) -> tensor<4x2xf32>
  %4 = "mhlo.concatenate"(%2, %3) {dimension = 1 : i64} : (tensor<4x5xf32>, tensor<4x2xf32>) -> tensor<4x7xf32>
  return %4 : tensor<4x7xf32>
}

// CHECK-LABEL: func.func @slice_reshape_slice_concat
// CHECK-NEXT: mhlo.fusion
// CHECK-NEXT:  mhlo.slice
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.slice
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.concatenate
// CHECK-NEXT:  mhlo.return
// CHECK:  {__byteir_elementwise_fusion__}
// CHECK: return
