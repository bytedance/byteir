func.func @add(%arg0 : tensor<128x2xf32>, %arg1 : tensor<128x2xf32>) -> tensor<128x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<128x2xf32>
  func.return %0 : tensor<128x2xf32>
}
