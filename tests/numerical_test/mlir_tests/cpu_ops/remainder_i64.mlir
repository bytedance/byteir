func.func @remainder(%arg0 : tensor<256x1xi64>) -> tensor<256x1xi64> { 
  %0 = stablehlo.constant dense<86400> : tensor<256x1xi64>
  %1 = "stablehlo.remainder"(%arg0, %0) : (tensor<256x1xi64>, tensor<256x1xi64>) -> tensor<256x1xi64>
  func.return %1 : tensor<256x1xi64>
}
