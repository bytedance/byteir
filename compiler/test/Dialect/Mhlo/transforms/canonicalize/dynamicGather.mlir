// RUN: byteir-opt %s --canonicalize | FileCheck %s

func.func @canonicalize_dynamic_gather(%arg0: tensor<375682x256xf16>, %arg1: tensor<16x64xi64>) -> tensor<16x64x256xf16> {
  %29 = "arith.constant"() {value = dense<[1, 256]> : tensor<2xi64>} : () -> tensor<2xi64>
  %143 = "mhlo.dynamic_gather"(%arg0, %arg1, %29) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<375682x256xf16>, tensor<16x64xi64>, tensor<2xi64>) -> tensor<16x64x256xf16>
  return %143 : tensor<16x64x256xf16>
}
// CHECK-NOT: mhlo.dynamic_gather
// CHECK: "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 256]> : tensor<2xi64>} : (tensor<375682x256xf16>, tensor<16x64xi64>) -> tensor<16x64x256xf16>
