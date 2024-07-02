func.func @convert_i16_f16(%arg0 : tensor<1x256xi16>) -> tensor<1x256xf16> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xi16>) -> tensor<1x256xf16>
  func.return %0 : tensor<1x256xf16>
}