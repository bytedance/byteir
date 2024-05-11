func.func @select_f64(%pred : tensor<256x1xi1>, %on_true : tensor<256x1xf64>, %on_false : tensor<256x1xf64>) -> tensor<256x1xf64> {
  %0 = "stablehlo.select"(%pred, %on_true, %on_false) : (tensor<256x1xi1>, tensor<256x1xf64>, tensor<256x1xf64>) -> tensor<256x1xf64>
  func.return %0 : tensor<256x1xf64>
}
