func.func @maximum_f64(%arg0 : tensor<256x1xf64>, %arg1 : tensor<256x1xf64>) -> tensor<256x1xf64> {
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<256x1xf64>, tensor<256x1xf64>) -> tensor<256x1xf64>
  func.return %0 : tensor<256x1xf64>
}
