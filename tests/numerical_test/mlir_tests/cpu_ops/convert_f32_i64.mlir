func.func @convert_f32_i64(%arg0 : tensor<1x256x1024xf32>) -> tensor<1x256x1024xi64> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xf32>) -> tensor<1x256x1024xi64>
  func.return %0 : tensor<1x256x1024xi64>
}