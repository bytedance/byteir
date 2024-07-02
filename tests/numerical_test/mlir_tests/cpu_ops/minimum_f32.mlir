func.func @minimum(%arg0 : tensor<256x1xf32>, %arg1 : tensor<256x1xf32>) -> tensor<256x1xf32> {
  %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<256x1xf32>, tensor<256x1xf32>) -> tensor<256x1xf32>
  func.return %0 : tensor<256x1xf32>
}
