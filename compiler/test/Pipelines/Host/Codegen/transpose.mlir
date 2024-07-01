// RUN: byteir-opt %s --insert-tile-and-vectorize-transpose-transform --transform-dialect-interpreter --split-input-file | FileCheck %s

func.func @transpose_nopad(%arg0: tensor<1x32x64x64xf32>) -> tensor<1x64x64x32xf32> {
  %0 = tensor.empty() : tensor<1x64x64x32xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<1x32x64x64xf32>) outs(%0 : tensor<1x64x64x32xf32>) permutation = [0, 2, 3, 1]
  // CHECK: vector.transpose
  return %1 : tensor<1x64x64x32xf32>
}

// CHECK-LABEL: transform.sequence
//   CHECK: transform.structured.tile
//   CHECK-NOT: transform.structured.pad
//   CHECK: transform.structured.vectorize

// -----
func.func @transpose_pad(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x3x224x224xf32> {
  %0 = tensor.empty() : tensor<1x3x224x224xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<1x224x224x3xf32>) outs(%0 : tensor<1x3x224x224xf32>) permutation = [0, 3, 1, 2]
  // CHECK-NOT: tensor.pad
  // CHECK: vector.transpose
  return %1 : tensor<1x3x224x224xf32>
}

// CHECK-LABEL: transform.sequence
//   CHECK: transform.structured.tile
//   CHECK: transform.structured.pad
//   CHECK: transform.structured.vectorize


// -----
func.func @transpose_split(%arg0: tensor<11x15x17x13xf32>) -> tensor<11x13x15x17xf32> {
  %0 = tensor.empty() : tensor<11x13x15x17xf32>
  %1 = linalg.transpose ins(%arg0 : tensor<11x15x17x13xf32>) outs(%0 : tensor<11x13x15x17xf32>) permutation = [0, 3, 1, 2]
  return %1 : tensor<11x13x15x17xf32>
  // CHECK: vector.transpose
}

// CHECK-LABEL: transform.sequence
//   CHECK: transform.structured.split
//   CHECK: merge_handle
//   CHECK: transform.structured.split
//   CHECK: merge_handle
//   CHECK: transform.structured.tile
//   CHECK: transform.structured.pad
//   CHECK: transform.structured.vectorize
