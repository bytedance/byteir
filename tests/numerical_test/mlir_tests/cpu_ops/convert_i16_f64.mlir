func.func @convert_i16_f64(%arg0 : tensor<1x256xi16>) -> tensor<1x256xf64> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xi16>) -> tensor<1x256xf64>
  func.return %0 : tensor<1x256xf64>
}