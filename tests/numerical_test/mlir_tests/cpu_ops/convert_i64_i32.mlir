func.func @convert_i64_i32(%arg0 : tensor<1x256xi64>) -> tensor<1x256xi32> { 
  %0 = stablehlo.convert %arg0 : (tensor<1x256xi64>) -> tensor<1x256xi32>
  func.return %0 : tensor<1x256xi32>
}