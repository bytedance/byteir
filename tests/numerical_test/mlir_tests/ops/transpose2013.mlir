func.func @transpose2013(%arg0 : tensor<12x64x256x128xf32>) -> tensor<256x12x64x128xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[2, 0, 1, 3]> : tensor<4xi64>} : (tensor<12x64x256x128xf32>) -> tensor<256x12x64x128xf32>
  return %0 : tensor<256x12x64x128xf32>
}
