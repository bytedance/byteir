func.func @convert_f64_i16(%arg0 : tensor<1x256xf64>) -> tensor<1x256xi16> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xf64>) -> tensor<1x256xi16>
  func.return %0 : tensor<1x256xi16>
}