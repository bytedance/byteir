func.func @bmm_rrr(%arg0 : tensor<12x256x256xf32>, %arg1 : tensor<12x256x64xf32>) -> tensor<1x256x12x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<12x256x256xf32>, tensor<12x256x64xf32>) -> tensor<12x256x64xf32>
    %1 = mhlo.reshape %0 : (tensor<12x256x64xf32>) -> tensor<1x12x256x64xf32>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x12x256x64xf32>) -> tensor<1x256x12x64xf32>
    return %2 : tensor<1x256x12x64xf32>
}
