func.func @transpose(%arg0 : tensor<1x12x64x256xf32>) -> tensor<1x256x12x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<1x12x64x256xf32>) -> tensor<1x256x12x64xf32>
  return %0 : tensor<1x256x12x64xf32>
}
