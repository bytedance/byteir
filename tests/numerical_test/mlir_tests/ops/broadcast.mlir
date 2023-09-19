func.func @broadcast(%arg0 : tensor<1x1x256x128xf16>) -> tensor<1x32x256x128xf16> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x256x128xf16>) -> tensor<1x32x256x128xf16>
  return %0 : tensor<1x32x256x128xf16>
}
