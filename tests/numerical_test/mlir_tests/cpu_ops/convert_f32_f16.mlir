func.func @convert_f32_f16(%arg0 : tensor<1x256x1024xf32>) -> tensor<1x256x1024xf16> {
  %0 = mhlo.convert %arg0 : (tensor<1x256x1024xf32>) -> tensor<1x256x1024xf16>
  return %0 : tensor<1x256x1024xf16>
}
