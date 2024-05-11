func.func @convert_f32_i32(%arg0 : tensor<1x256x1024xf32>) -> tensor<1x256x1024xi32> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xf32>) -> tensor<1x256x1024xi32>
  func.return %0 : tensor<1x256x1024xi32>
}