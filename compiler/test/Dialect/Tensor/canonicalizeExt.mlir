// RUN: byteir-opt %s -canonicalize-ext="blind-fold" --split-input-file | FileCheck %s

// CHECK-LABEL: func @slice_constant_3x4_offsets
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[CONST:.+]] = arith.constant dense<{{\[}}[1.200000e+01, 1.300000e+01], [3.000000e+00, 5.000000e+00]]> : tensor<2x2xf32>
//       CHECK:   return %[[CONST]] :  tensor<2x2xf32>
func.func @slice_constant_3x4_offsets(%arg0 : tensor<3x4xf32>) -> tensor<2x2xf32>
{
  %cst = arith.constant dense<[[10.0, 9.0, 8.0, 7.0], [11.0, 12.0, 13.0, 14.0], [1.0, 3.0, 5.0, 7.0]]> : tensor<3x4xf32>
  %slice = tensor.extract_slice %cst[1, 1] [2, 2] [1, 1] : tensor<3x4xf32> to tensor<2x2xf32>
  return %slice : tensor<2x2xf32>
}

// ----

func.func @fold_rank_reduced_extract_slice_and_collapse_shape(%arg0: tensor<19x1024x1xi32>) -> tensor<1024xi32> {
  %0 = tensor.extract_slice %arg0[0, 0, 0][1, 1024, 1][1, 1, 1] : tensor<19x1024x1xi32> to tensor<1x1024x1xi32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<1x1024x1xi32> into tensor<1024xi32>
  func.return %1 : tensor<1024xi32>
}
// CHECK-LABEL: fold_rank_reduced_extract_slice_and_collapse_shape
//   CHECK-NEXT: tensor.extract_slice
//     CHECK-SAME: tensor<19x1024x1xi32> to tensor<1024xi32>
//   CHECK-NOT: tensor.collapse_shape

// ----

func.func @extract_slice_and_collapse_shape_no_fold(%arg0: tensor<19x1024x1xi32>) -> tensor<2048xi32> {
  %0 = tensor.extract_slice %arg0[0, 0, 0][2, 1024, 1][1, 1, 1] : tensor<19x1024x1xi32> to tensor<2x1024x1xi32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<2x1024x1xi32> into tensor<2048xi32>
  func.return %1 : tensor<2048xi32>
}
// CHECK-LABEL: extract_slice_and_collapse_shape_no_fold
//   CHECK: tensor.extract_slice
//   CHECK: tensor.collapse_shape

// ----

func.func @fold_zero_rank_from_elements_insert_slice(%arg0: tensor<1024xf32>, %scalar : f32) -> tensor<1024xf32> {
  %0 = tensor.from_elements %scalar : tensor<f32>
  %1 = tensor.insert_slice %0 into %arg0[256] [1] [1] : tensor<f32> into tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
// CHECK-LABEL: fold_zero_rank_from_elements_insert_slice
//   CHECK: tensor.insert
//   CHECK-NOT: tensor.from_elements
//   CHECK-NOT: tensor.insert_slice