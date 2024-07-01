func.func @convert_f32_f64(%arg0 : tensor<1x256xf32>) -> tensor<1x256xf64> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xf32>) -> tensor<1x256xf64>
  func.return %0 : tensor<1x256xf64>
}