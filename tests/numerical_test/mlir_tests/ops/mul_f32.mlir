func.func @multiply_f32(%arg0 : tensor<1x32x256x128xf32>, %arg1 : tensor<1x32x256x128xf32>) -> tensor<1x32x256x128xf32> {
  %0 = mhlo.multiply %arg0, %arg1 : tensor<1x32x256x128xf32>
  return %0 : tensor<1x32x256x128xf32>
}
