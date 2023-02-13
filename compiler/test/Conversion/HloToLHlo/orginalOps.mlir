// RUN: byteir-opt -convert-hlo-to-lhlo %s | FileCheck %s
  
func.func @aten__index_select.148(%arg0: tensor<30522x128xf32>, %arg1: tensor<128xui32>) -> tensor<128x128xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
// CHECK: lmhlo.gather

func.func private @xla__select.55(%arg0: tensor<1x512xi64>) -> tensor<1x128xi64> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
  return %0 : tensor<1x128xi64>
}
// CHECK: lmhlo.slice

func.func @mhlo_reduce(%arg0: tensor<1x128x128xf32>) -> tensor<128xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %1 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
  return %1: tensor<128xf32>
}
// CHECK: lmhlo.reduce

func.func @mhlo_convert(%arg0: tensor<128x?xf32>) -> tensor<128x?xf16> {
  %0 = "mhlo.convert"(%arg0) : (tensor<128x?xf32>) -> tensor<128x?xf16>
  return %0: tensor<128x?xf16>
}
// CHECK: lmhlo.convert