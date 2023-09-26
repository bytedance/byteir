func.func @bmm_rrr_permute(%arg0: tensor<32x64x64xf16>, %arg1: tensor<32x64x128xf16>) -> tensor<1x64x32x128xf16> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<32x64x64xf16>, tensor<32x64x128xf16>) -> tensor<32x64x128xf16>
    %1 = mhlo.reshape %0 : (tensor<32x64x128xf16>) -> tensor<1x32x64x128xf16>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<1x32x64x128xf16>) -> tensor<1x64x32x128xf16>
    return %2 : tensor<1x64x32x128xf16>
}