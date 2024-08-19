func.func @bmm_rcr(%arg0 : tensor<32x256x128xf16>, %arg1 : tensor<32x256x128xf16>) -> tensor<32x256x256xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<32x256x128xf16>) -> tensor<32x128x256xf16>
    %1 = "mhlo.dot_general"(%arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<32x256x128xf16>, tensor<32x128x256xf16>) -> tensor<32x256x256xf16>
    return %1 : tensor<32x256x256xf16>
}
