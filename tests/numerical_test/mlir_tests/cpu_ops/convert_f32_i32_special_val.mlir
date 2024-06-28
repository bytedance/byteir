func.func @convert_f32_i32_special_val(%arg0 : tensor<2x3xf32>) -> tensor<2x3xi32> { 
  %0 = stablehlo.convert %arg0 : (tensor<2x3xf32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}