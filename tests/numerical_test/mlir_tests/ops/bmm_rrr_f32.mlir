func.func @bmm_rrr(%arg0 : tensor<32x256x256xf32>, %arg1 : tensor<32x256x128xf32>) -> tensor<32x256x128xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<32x256x256xf32>, tensor<32x256x128xf32>) -> tensor<32x256x128xf32>
    return %0 : tensor<32x256x128xf32>
}
