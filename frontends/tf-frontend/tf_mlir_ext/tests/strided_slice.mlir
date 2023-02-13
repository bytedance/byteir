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
