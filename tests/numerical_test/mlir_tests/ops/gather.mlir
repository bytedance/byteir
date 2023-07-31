func.func @gather(%arg0 : tensor<256x128xf16>, %arg1 : tensor<1x256xi64>) -> tensor<1x256x128xf16> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<256x128xf16>, tensor<1x256xi64>) -> tensor<1x256x128xf16>
  return %0 : tensor<1x256x128xf16>
}

