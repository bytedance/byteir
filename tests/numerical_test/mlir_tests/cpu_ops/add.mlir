func.func @add(%arg0 : tensor<256x256xf32>, %arg1 : tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<256x256xf32>
  func.return %0 : tensor<256x256xf32>
}
