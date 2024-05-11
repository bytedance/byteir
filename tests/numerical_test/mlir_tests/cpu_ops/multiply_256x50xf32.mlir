func.func @multiply_256x50xf32(%arg0 : tensor<256x50xf32>, %arg1 : tensor<256x50xf32>) -> tensor<256x50xf32> {
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<256x50xf32>, tensor<256x50xf32>) -> tensor<256x50xf32>
  func.return %0 : tensor<256x50xf32>
}
