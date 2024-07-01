// RUN: byteir-opt -tensor-to-linalg -split-input-file %s | FileCheck %s


func.func @expand_shape_static(%arg0: tensor<1000xf32>) -> tensor<1x1000xf32> {
  %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 1000] : tensor<1000xf32> into tensor<1x1000xf32>
  return %expanded : tensor<1x1000xf32>
}
// CHECK-LABEL: @expand_shape_static
// CHECK: linalg.generic

func.func @collapse_shape_static(%arg0: tensor<1x3x4x1x5xf32>) -> tensor<3x4x5xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2], [3, 4]] :
    tensor<1x3x4x1x5xf32> into tensor<3x4x5xf32>
  return %0 : tensor<3x4x5xf32>
}
// CHECK-LABEL: @collapse_shape_static
// CHECK: linalg.generic
