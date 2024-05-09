func.func @convert_i16_i32(%arg0 : tensor<1x256x1024xi16>) -> tensor<1x256x1024xi32> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xi16>) -> tensor<1x256x1024xi32>
  func.return %0 : tensor<1x256x1024xi32>
}