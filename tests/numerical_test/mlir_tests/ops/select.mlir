func.func @select(%arg0 : tensor<1x32x256x256xi1>, %arg1 : tensor<1x32x256x256xf32>, %arg2 : tensor<1x32x256x256xf32>) -> tensor<1x32x256x256xf32> {
  %0 = mhlo.select %arg0, %arg1, %arg2 : tensor<1x32x256x256xi1>, tensor<1x32x256x256xf32>
  return %0 : tensor<1x32x256x256xf32>
}
