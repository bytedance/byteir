// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true" | FileCheck %s

func.func @dead_custom_call() -> tensor<128xf32> {
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
  %1 = "mhlo.custom_call"(%c0) {backend_config = "", call_target_name = "foo", has_side_effect = false} : (tensor<128xf32>) -> tensor<128xf32>
  return %c0: tensor<128xf32>
}
// CHECK-LABEL: dead_custom_call
// CHECK-NOT: mhlo.custom_call

func.func @remove_empty(%arg0: tensor<150x768xf16>) -> tensor<150x768xf16> {
  call @empty() : () -> ()
  return %arg0 : tensor<150x768xf16>
}
func.func private @empty() {
  return
}
// CHECK-LABEL: remove_empty
// CHECK-NOT: call @empty

func.func @fold_useless_shape_broadcast(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = shape.const_shape [4] : tensor<1xindex>
  %1 = mhlo.constant dense<[[-0.570340514, 0.117151208, -0.135694504, -1.57919896], [0.520053327, 0.762166619, 0.322875232, -1.69871449], [-1.26622009, 0.63558042, 5.698780e-01, 0.954656243], [0.776482939, 0.348752886, 2.03235912, 0.837243676]]> : tensor<4x4xf32>
  %2 = "mhlo.dot"(%arg0, %1) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %3 = shape.shape_of %2 : tensor<?x4xf32> -> tensor<2xindex>
  %4 = shape.broadcast %3, %0 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%2, %4) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  return %5 : tensor<?x4xf32>
}
// CHECK-LABEL: fold_useless_shape_broadcast
// CHECK-NOT: shape.broadcast


func.func @canonicalize_dynamic_conv_case0(%1212: tensor<?x10x19x4xf16>, %85: tensor<5x7x4x12xf16>) -> tensor<?x6x13x12xf16> {
  %cst_1 = arith.constant dense<0> : tensor<4xi32>
  %1214 = "mhlo.dynamic_conv"(%1212, %85, %cst_1) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<?x10x19x4xf16>, tensor<5x7x4x12xf16>, tensor<4xi32>) -> tensor<?x6x13x12xf16>
  return %1214 : tensor<?x6x13x12xf16>
}
// CHECK-LABEL: canonicalize_dynamic_conv_case0
// CHECK-NOT:  mhlo.dynamic_conv
// CHECK:  mhlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]{{\]}}, rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x10x19x4xf16>, tensor<5x7x4x12xf16>) -> tensor<?x6x13x12xf16>

func.func @canonicalize_dynamic_conv_case1(%1212: tensor<?x10x19x4xf16>, %85: tensor<5x7x4x12xf16>) -> tensor<?x8x15x12xf16> {
  %cst_1 = arith.constant dense<1> : tensor<4xi32>
  %1214 = "mhlo.dynamic_conv"(%1212, %85, %cst_1) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<?x10x19x4xf16>, tensor<5x7x4x12xf16>, tensor<4xi32>) -> tensor<?x8x15x12xf16>
  return %1214 : tensor<?x8x15x12xf16>
}
// CHECK-LABEL: canonicalize_dynamic_conv_case1
// CHECK-NOT: mhlo.dynamic_conv
// CHECK:  mhlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[}}[1, 1], [1, 1]{{\]}}, rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x10x19x4xf16>, tensor<5x7x4x12xf16>) -> tensor<?x8x15x12xf16>


func.func @slice_fold_large_outputs() -> tensor<999998xi64> {
  %0 = mhlo.constant dense<1> : tensor<1000000xi64>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[999999]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1000000xi64>) -> (tensor<999998xi64>)
  func.return %1 : tensor<999998xi64>
}
// CHECK-LABEL: slice_fold_large_outputs
// CHECK-NOT: mhlo.slice

func.func @add_insert_slices(%arg0: tensor<64x256x384xf32>, %arg1: tensor<64x256x384xf32>, %arg2: tensor<64x256x384xf32>) -> tensor<64x256x1152xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<64x256x1152xf32>
  %inserted_slice = tensor.insert_slice %arg0 into %0[0, 0, 768] [64, 256, 384] [1, 1, 1] : tensor<64x256x384xf32> into tensor<64x256x1152xf32>
  %inserted_slice_0 = tensor.insert_slice %arg1 into %0[0, 0, 384] [64, 256, 384] [1, 1, 1] : tensor<64x256x384xf32> into tensor<64x256x1152xf32>
  %1 = mhlo.add %inserted_slice, %inserted_slice_0 : tensor<64x256x1152xf32>
  %inserted_slice_1 = tensor.insert_slice %arg2 into %0[0, 0, 0] [64, 256, 384] [1, 1, 1] : tensor<64x256x384xf32> into tensor<64x256x1152xf32>
  %2 = mhlo.add %1, %inserted_slice_1 : tensor<64x256x1152xf32>
  return %2 : tensor<64x256x1152xf32>
}
// CHECK-LABEL: add_insert_slices
// CHECK: tensor.insert_slice
// CHECK: tensor.insert_slice
// CHECK: tensor.insert_slice
// CHECK-NOT: mhlo.add
