func.func @bmm_rrr_add(%arg0 : tensor<32x256x256xf16>, %arg1 : tensor<32x256x128xf16>, %arg2 : tensor<1x32x256x128xf16>) -> tensor<1x32x256x128xf16> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<32x256x256xf16>, tensor<32x256x128xf16>) -> tensor<32x256x128xf16>
    %1 = mhlo.reshape %0 : (tensor<32x256x128xf16>) -> tensor<1x32x256x128xf16>
    %2 = mhlo.add %arg2, %1 : tensor<1x32x256x128xf16>
    return %2 : tensor<1x32x256x128xf16>
}
