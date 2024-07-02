func.func @convert_f32_i32(%arg0 : tensor<1x256xf32>) -> tensor<1x256xi32> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xf32>) -> tensor<1x256xi32>
  func.return %0 : tensor<1x256xi32>
}