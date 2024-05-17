// RUN: tf-ext-opt -mhlo-legalize-tf-ext %s | FileCheck %s

func.func @dynamic_strided_slice(%arg0: tensor<?x26xf16>) -> tensor<?x12xf16> {
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 3 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x26xf16>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x12xf16>
  return %3 : tensor<?x12xf16> 
}
// CHECK-LABEL: dynamic_strided_slice
// CHECK: mhlo.real_dynamic_slice

func.func @static_strided_slice(%arg0: tensor<1x26xf16>) -> tensor<1x12xf16> {
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<[0, 12]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tf.StridedSlice"(%arg0, %0, %1, %2) { begin_mask = 3 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1x26xf16>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x12xf16>
  return %3 : tensor<1x12xf16> 
}
// CHECK-LABEL: static_strided_slice
// CHECK: "tf.StridedSlice"

func.func @static_batch_matmul_v2(%arg0: tensor<32x246x16xf16>, %arg1: tensor<32x16x16xf16>) -> tensor<32x246x16xf16> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false, device = ""} : (tensor<32x246x16xf16>, tensor<32x16x16xf16>) -> tensor<32x246x16xf16>
  return %0 : tensor<32x246x16xf16> 
}
// CHECK-LABEL: @static_batch_matmul_v2
// CHECK-NEXT: %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x246x16xf16>, tensor<32x16x16xf16>) -> tensor<32x246x16xf16>
// CHECK-NEXT: return %0 : tensor<32x246x16xf16>

func.func @round_fp32(%arg0: tensor<6xf32>) -> tensor<6xf32> {
  %0 = "tf.Round"(%arg0) : (tensor<6xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}
// CHECK-LABEL: @round_fp32
// CHECK-NEXT: %0 = mhlo.round_nearest_even %arg0 : tensor<6xf32>
// CHECK-NEXT: return %0 : tensor<6xf32>

func.func @round_int32(%arg0: tensor<6xi32>) -> tensor<6xi32> {
  %0 = "tf.Abs"(%arg0) : (tensor<6xi32>) -> tensor<6xi32>
  %1 = "tf.Round"(%0) : (tensor<6xi32>) -> tensor<6xi32>
  return %1 : tensor<6xi32>
}
// CHECK-LABEL: @round_int32
// CHECK-NEXT: %0 = "tf.Abs"(%arg0) : (tensor<6xi32>) -> tensor<6xi32>
// CHECK-NEXT: return %0 : tensor<6xi32>
