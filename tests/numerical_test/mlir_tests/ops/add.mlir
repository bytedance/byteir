func.func @add(%arg0 : tensor<256x256xf32>, %arg1 : tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}

