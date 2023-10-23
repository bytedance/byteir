func.func @transpose120(%arg0 : tensor<12x64x256xf32>) -> tensor<64x256x12xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 2, 0]> : tensor<3xi64>} : (tensor<12x64x256xf32>) -> tensor<64x256x12xf32>
  return %0 : tensor<64x256x12xf32>
}