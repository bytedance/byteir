func.func @negate(%arg0 : tensor<1x256xf32>) -> tensor<1x256xf32> {
  %0 = mhlo.negate %arg0 : tensor<1x256xf32>
  return %0 : tensor<1x256xf32>
}
