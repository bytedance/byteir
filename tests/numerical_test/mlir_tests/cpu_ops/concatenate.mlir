func.func @concatenate(%arg0 : tensor<256x1xi64>) -> tensor<256x2xi64> { 
  %0 = stablehlo.constant dense<86400> : tensor<256x1xi64>
  %1 = "stablehlo.concatenate"(%0, %arg0) {
    dimension = 1 : i64,
    device = "host"
  } : (tensor<256x1xi64>, tensor<256x1xi64>) -> tensor<256x2xi64>
  func.return %1 : tensor<256x2xi64>
}
