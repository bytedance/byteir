func.func @convert_f16_i64(%arg0 : tensor<1x256x1024xf16>) -> tensor<1x256x1024xi64> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xf16>) -> tensor<1x256x1024xi64>
  func.return %0 : tensor<1x256x1024xi64>
}