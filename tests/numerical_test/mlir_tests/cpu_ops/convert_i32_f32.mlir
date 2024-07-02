func.func @convert_i32_f32(%arg0 : tensor<1x256xi32>) -> tensor<1x256xf32> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xi32>) -> tensor<1x256xf32>
  func.return %0 : tensor<1x256xf32>
}