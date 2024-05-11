func.func @maximum_i32(%arg0 : tensor<256x50xi32>, %arg1 : tensor<256x50xi32>) -> tensor<256x50xi32> {
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<256x50xi32>, tensor<256x50xi32>) -> tensor<256x50xi32>
  func.return %0 : tensor<256x50xi32>
}
