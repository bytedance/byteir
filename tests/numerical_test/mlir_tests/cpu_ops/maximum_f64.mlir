func.func @maximum_f64(%arg0 : tensor<256x50xf64>, %arg1 : tensor<256x50xf64>) -> tensor<256x50xf64> {
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<256x50xf64>, tensor<256x50xf64>) -> tensor<256x50xf64>
  func.return %0 : tensor<256x50xf64>
}
