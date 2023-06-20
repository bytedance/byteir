// RUN: byteir-opt %s -test-mhlo-canonicalize-ext | FileCheck %s

func.func @dead_custom_call() -> tensor<128xf32> {
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<128xf32>
  %1 = "mhlo.custom_call"(%c0) {backend_config = "", call_target_name = "foo", has_side_effect = false} : (tensor<128xf32>) -> tensor<128xf32>
  return %c0: tensor<128xf32>
}
// CHECK-LABEL: dead_custom_call
// CHECK-NOT: mhlo.custom_call

func.func @eliminate_splat_constant_transpose() -> tensor<2x1x4x3xi32> {
  %0 = mhlo.constant dense<0> : tensor<1x2x3x4xi32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %1: tensor<2x1x4x3xi32>
}
// CHECK-LABEL: eliminate_splat_constant_transpose
// CHECK-NEXT: %0 = mhlo.constant dense<0> : tensor<2x1x4x3xi32>

func.func @transpose_non_splat_constant_2d() -> tensor<2x1xf32> {
  %0 = mhlo.constant dense<[[1.0000, 0.0000]]> : tensor<1x2xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
// CHECK-LABEL: transpose_non_splat_constant_2d
// CHECK{LITERAL}:  mhlo.constant dense<[[1.000000e+00], [0.000000e+00]]> : tensor<2x1xf32>

func.func @transpose_non_splat_constant_3d() -> tensor<2x2x2xf32> {
  %0 = mhlo.constant dense<[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]> : tensor<2x2x2xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[2, 1, 0]> : tensor<3xi64>} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  return %1 : tensor<2x2x2xf32>
}
// CHECK-LABEL: transpose_non_splat_constant_3d
// CHECK{LITERAL}:  mhlo.constant dense<[[[0.000000e+00, 4.000000e+00], [2.000000e+00, 6.000000e+00]], [[1.000000e+00, 5.000000e+00], [3.000000e+00, 7.000000e+00]]]> : tensor<2x2x2xf32>

// FIXME: make constant really large or trigger canonicalize-ext anywhy.
func.func @fold_large_constant_binary_op() -> tensor<2xf32> {
  %0 = mhlo.constant dense<[0.00000e+0, 1.00000e+0]> : tensor<2xf32>
  %1 = mhlo.constant dense<[1.00000e+0, 1.00000e+0]> : tensor<2xf32>
  %2 = mhlo.add %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}
// CHECK-LABEL: fold_large_constant_binary_op
// CHECK-NOT: mhlo.add
// CHECK: mhlo.constant dense<[1.000000e+00, 2.000000e+00]>

func.func @fold_concat_of_continuous_slices(%arg0: tensor<4x11xf32>) -> tensor<4x11xf32> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 7]> : tensor<2xi64>, start_indices = dense<[0, 5]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x2xf32>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 11]> : tensor<2xi64>, start_indices = dense<[0, 7]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x4xf32>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x11xf32>) -> tensor<4x5xf32>
  %3 = "mhlo.concatenate"(%2, %0, %1) {dimension = 1 : i64} : (tensor<4x5xf32>, tensor<4x2xf32>, tensor<4x4xf32>) -> tensor<4x11xf32>
  return %3 : tensor<4x11xf32> 
}
// CHECK-LABEL: func.func @fold_concat_of_continuous_slices
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x11xf32>)
// CHECK-NEXT: return %[[ARG0]] : tensor<4x11xf32>

func.func @not_fold_concat_of_slice(%655: tensor<1x112x56x128xf16>) -> tensor<1x56x112x128xf16> {
  %656 = "mhlo.slice"(%655) {limit_indices = dense<[1, 59, 56, 128]> : tensor<4xi64>, start_indices = dense<[0, 3, 0, 0]> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<1x112x56x128xf16>) -> tensor<1x56x56x128xf16>
  %657 = "mhlo.concatenate"(%656, %656) {dimension = 2 : i64} : (tensor<1x56x56x128xf16>, tensor<1x56x56x128xf16>) -> tensor<1x56x112x128xf16>
  func.return %657 : tensor<1x56x112x128xf16>
}
// CHECK-LEBEL: func.func @not_fold_concat_of_slice
// CHECK:  "mhlo.slice"
// CHECK:  "mhlo.concatenate"

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

func.func @concat_const_folding_with_dup_value() -> tensor<3x2xi64> {
  %0 = mhlo.constant dense<1> : tensor<1x2xi64>
  %1 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
  %3 = "mhlo.concatenate"(%0, %1, %1) {dimension = 0 : i64} : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<3x2xi64>
  return %3 : tensor<3x2xi64>
}
// CHECK-LABEL: concat_const_folding_with_dup_value
// CHECK: %0 = mhlo.constant dense<{{\[}}[1, 1], [2, 3], [2, 3]{{\]}}> : tensor<3x2xi64>
// CHECK: return %0 : tensor<3x2xi64>

func.func @concat_const_folding_case1(%arg0: tensor<1x2xi64>) -> tensor<5x2xi64> {
  %0 = mhlo.constant dense<1> : tensor<1x2xi64>
  %1 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
  %2 = mhlo.constant dense<[[3, 4]]> : tensor<1x2xi64>
  %4 = "mhlo.concatenate"(%0, %1, %arg0, %2, %2) {dimension = 0 : i64} : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<5x2xi64>
  return %4 : tensor<5x2xi64>
}
// CHECK-LABEL: concat_const_folding_case1
// CHECK:  %0 = mhlo.constant dense<{{\[}}[1, 1], [2, 3]{{\]}}> : tensor<2x2xi64>
// CHECK:  %1 = mhlo.constant dense<{{\[}}[3, 4], [3, 4]{{\]}}> : tensor<2x2xi64>

func.func @canonicalize_const_broadcast() -> tensor<1x10x2xi64> {
  %0 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi64>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x2xi64>) -> (tensor<1x10x2xi64>)
  return %1 : tensor<1x10x2xi64>
}
// CHECK-LABEL: canonicalize_const_broadcast
// CHECK: %[[V0:.*]] = mhlo.constant dense<[2, 3]> : tensor<2xi64>
// CHECK: "mhlo.broadcast_in_dim"(%[[V0]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1x10x2xi64>

func.func @fold_clamp() -> tensor<5xi64> {
  %1 = mhlo.constant dense<[-1, 100, 200, 0, 149]> : tensor<5xi64>
  %2 = mhlo.constant dense<149> : tensor<i64>
  %3 = mhlo.constant dense<0> : tensor<i64>
  %4 = mhlo.clamp %3, %1, %2 : (tensor<i64>, tensor<5xi64>, tensor<i64>) -> tensor<5xi64>
  return %4 : tensor<5xi64>
}
// CHECK-LABEL: fold_clamp
// CHECK: mhlo.constant dense<[0, 100, 149, 0, 149]> : tensor<5xi64>
// CHECK-NOT: mhlo.clamp

func.func @clamp_fold() -> tensor<5xi64> {
  %0 = mhlo.constant dense<[149, 101, -1,  30, 50]> : tensor<5xi64>
  %1 = mhlo.constant dense<[-1,  100, 200, 0,  149]> : tensor<5xi64>
  %2 = mhlo.constant dense<[0,   10,  -10, 10, -100]> : tensor<5xi64>
  %3 = mhlo.clamp %2, %1, %0 : (tensor<5xi64>, tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
  return %3 : tensor<5xi64>
}
// CHECK-LABEL: clamp_fold
// CHECK{LITERAL}: mhlo.constant dense<[0, 100, -1, 10, 50]>
// CHECK-NOT: mhlo.clamp

func.func @clamp_fold_float() -> tensor<6xf32> {
  %0 = mhlo.constant dense<[5.0, 66.0, 0xFFFFFFFF, -2.0,       0xFFFFFFFF, 6.0]> : tensor<6xf32>
  %1 = mhlo.constant dense<[5.0, 3.0,  2.0,        0xFFFFFFFF, 0xFFFFFFFF, 4.0]> : tensor<6xf32>
  %2 = mhlo.constant dense<[5.0, 1.0,  1.0,        0xFFFFFFFF, 0xFFFFFFFF, 5.0]> : tensor<6xf32>
  %3 = mhlo.clamp %2, %1, %0 : (tensor<6xf32>, tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %3 : tensor<6xf32>
}
// CHECK-LABEL: clamp_fold_float
// CHECK{LITERAL}: mhlo.constant dense<[5.000000e+00, 3.000000e+00, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 5.000000e+00]
// CHECK-NOT: mhlo.clamp

func.func @simplify_byteir_addn(%arg0: tensor<150x768xf16>, %arg1: tensor<150x768xf16>) -> tensor<150x768xf16> {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {api_version = 1 : i32, backend_config = "", byteir_attrs = {_grappler_ArithmeticOptimizer_AddOpsRewriteStage = true}, call_target_name = "byteir.addn", called_computations = [], has_side_effect = false, output_operand_aliases = []} : (tensor<150x768xf16>, tensor<150x768xf16>) -> tensor<150x768xf16>
  return %0 : tensor<150x768xf16>
}
// CHECK-LABEL: simplify_byteir_addn
// CHECK-NOT: mhlo.custom_call
// CHECK: mhlo.add

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
// CHECK: mhlo.concatenate
// CHECK-NOT: mhlo.add
