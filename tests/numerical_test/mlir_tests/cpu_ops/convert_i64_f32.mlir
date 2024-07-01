func.func @convert_i64_f32(%arg0 : tensor<1x256xi64>) -> tensor<1x256xf32> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xi64>) -> tensor<1x256xf32>
  func.return %0 : tensor<1x256xf32>
}