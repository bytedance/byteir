func.func @convert_f16_f32(%arg0 : tensor<1x256x1024xf16>) -> tensor<1x256x1024xf32> {
  %0 = mhlo.convert %arg0 : (tensor<1x256x1024xf16>) -> tensor<1x256x1024xf32>
  return %0 : tensor<1x256x1024xf32>
}
