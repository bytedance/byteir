func.func @convert_i32_f64(%arg0 : tensor<1x256x1024xi32>) -> tensor<1x256x1024xf64> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xi32>) -> tensor<1x256x1024xf64>
  func.return %0 : tensor<1x256x1024xf64>
}