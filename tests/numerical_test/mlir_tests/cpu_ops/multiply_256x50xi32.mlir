func.func @multiply_256x50xi32(%arg0 : tensor<256x50xi32>, %arg1 : tensor<256x50xi32>) -> tensor<256x50xi32> {
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<256x50xi32>, tensor<256x50xi32>) -> tensor<256x50xi32>
  func.return %0 : tensor<256x50xi32>
}
