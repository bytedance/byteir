func.func @convert_i64_f16(%arg0 : tensor<1x256xi64>) -> tensor<1x256xf16> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xi64>) -> tensor<1x256xf16>
  func.return %0 : tensor<1x256xf16>
}