func.func @convert_f64_i64(%arg0 : tensor<1x256xf64>) -> tensor<1x256xi64> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xf64>) -> tensor<1x256xi64>
  func.return %0 : tensor<1x256xi64>
}