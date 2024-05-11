func.func @convert_i32_i64(%arg0 : tensor<1x256x1024xi32>) -> tensor<1x256x1024xi64> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256x1024xi32>) -> tensor<1x256x1024xi64>
  func.return %0 : tensor<1x256x1024xi64>
}