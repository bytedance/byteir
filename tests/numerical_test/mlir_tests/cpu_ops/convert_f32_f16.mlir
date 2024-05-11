func.func @convert_f32_f16(%arg0 : tensor<1x256x1024xf32>) -> tensor<1x256x1024xf16> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xf32>) -> tensor<1x256x1024xf16>
  func.return %0 : tensor<1x256x1024xf16>
}