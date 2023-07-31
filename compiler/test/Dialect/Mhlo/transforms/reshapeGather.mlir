// RUN: byteir-opt %s -reshape-gather -canonicalize | FileCheck %s

func.func @gather_2d(%arg0 : tensor<1024x768xf32>, %arg1 : tensor<2x256xi64>) -> tensor<2x256x768xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 768]> : tensor<2xi64>} : (tensor<1024x768xf32>, tensor<2x256xi64>) -> tensor<2x256x768xf32>
  return %0 : tensor<2x256x768xf32>
}

// CHECK-LABEL: func.func @gather_2d
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.gather
// CHECK-NEXT:  mhlo.reshape

func.func @gather_3d(%arg0 : tensor<1024x256x768xf32>, %arg1 : tensor<2x256xi64>) -> tensor<2x256x256x768xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 256, 768]> : tensor<3xi64>} : (tensor<1024x256x768xf32>, tensor<2x256xi64>) -> tensor<2x256x256x768xf32>
  return %0 : tensor<2x256x256x768xf32>
}

// CHECK-LABEL: func.func @gather_3d
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.gather
// CHECK-NEXT:  mhlo.reshape
