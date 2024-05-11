func.func @convert_f64_i16(%arg0 : tensor<1x256x1024xf64>) -> tensor<1x256x1024xi16> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xf64>) -> tensor<1x256x1024xi16>
  func.return %0 : tensor<1x256x1024xi16>
}