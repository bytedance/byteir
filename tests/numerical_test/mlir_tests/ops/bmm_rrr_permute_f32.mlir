func.func @bmm_rrr_permute_f32(%arg0: tensor<4x2x2xf32>, %arg1: tensor<4x2x2xf32>) -> tensor<2x2x2x2xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x2x2xf32>, tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  %1 = mhlo.reshape %0 : (tensor<4x2x2xf32>) -> tensor<2x2x2x2xf32>
  %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %2 : tensor<2x2x2x2xf32>
}
