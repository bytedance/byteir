func.func @transpose1023(%arg0 : tensor<12x64x256x128xf32>) -> tensor<64x12x256x128xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 2, 3]> : tensor<4xi64>} : (tensor<12x64x256x128xf32>) -> tensor<64x12x256x128xf32>
  return %0 : tensor<64x12x256x128xf32>
}
