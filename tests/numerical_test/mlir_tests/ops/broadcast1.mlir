func.func @broadcast1(%arg0 : tensor<4096xf32>) -> tensor<1x256x4096xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<4096xf32>) -> tensor<1x256x4096xf32>
  return %0 : tensor<1x256x4096xf32>
}
