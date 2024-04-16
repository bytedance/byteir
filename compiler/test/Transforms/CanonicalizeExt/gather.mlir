// RUN: byteir-opt %s -canonicalize-ext="blind-fold=true" | FileCheck %s

func.func @replace_gather_with_input_0() -> (tensor<1x64x128xf16>, tensor<1x32x64x128xf16>) {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<64x128xf16>
  %1 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<64xi64>
  %2 = "mhlo.gather"(%0, %1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<64x128xf16>, tensor<64xi64>) -> tensor<64x128xf16>
  %3 = mhlo.reshape %2 : (tensor<64x128xf16>) -> tensor<1x64x128xf16>
  %4 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<64x128xf16>) -> tensor<1x32x64x128xf16>
  return %3, %4 : tensor<1x64x128xf16>, tensor<1x32x64x128xf16>
}
// CHECK-LABEL: @replace_gather_with_input_0
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: return

func.func @replace_gather_with_input_1(%arg0: tensor<64x128xf16>) -> (tensor<1x64x128xf16>, tensor<1x32x64x128xf16>) {
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<64xi64>
  %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<64x128xf16>, tensor<64xi64>) -> tensor<64x128xf16>
  %2 = mhlo.reshape %1 : (tensor<64x128xf16>) -> tensor<1x64x128xf16>
  %3 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<64x128xf16>) -> tensor<1x32x64x128xf16>
  return %2, %3 : tensor<1x64x128xf16>, tensor<1x32x64x128xf16>
}
// CHECK-LABEL: @replace_gather_with_input_1
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return

func.func @replace_gather_with_input_2(%arg0: tensor<64x128xf16>) -> (tensor<1x64x128xf16>, tensor<1x32x64x128xf16>) {
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<128xi64>
  %1 = "mhlo.gather"(%arg0, %0) {dimension_numbers = #mhlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[64, 1]> : tensor<2xi64>} : (tensor<64x128xf16>, tensor<128xi64>) -> tensor<64x128xf16>
  %2 = mhlo.reshape %1 : (tensor<64x128xf16>) -> tensor<1x64x128xf16>
  %3 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<64x128xf16>) -> tensor<1x32x64x128xf16>
  return %2, %3 : tensor<1x64x128xf16>, tensor<1x32x64x128xf16>
}
// CHECK-LABEL: @replace_gather_with_input_2
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: return
