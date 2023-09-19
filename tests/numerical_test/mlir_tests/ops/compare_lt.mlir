func.func @compare_lt(%arg0 : tensor<1x32x256x256xf32>, %arg1 : tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xi1> {
  %0 = mhlo.compare  LT, %arg0, %arg1,  FLOAT : (tensor<1x32x256x256xf32>, tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xi1>
  return %0 : tensor<1x32x256x256xi1>
}
