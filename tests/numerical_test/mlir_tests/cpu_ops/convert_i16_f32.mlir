func.func @convert_i16_f32(%arg0 : tensor<1x256xi16>) -> tensor<1x256xf32> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xi16>) -> tensor<1x256xf32>
  func.return %0 : tensor<1x256xf32>
}