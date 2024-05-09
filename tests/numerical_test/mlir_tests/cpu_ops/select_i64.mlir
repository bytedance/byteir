func.func @select_i64(%pred : tensor<256x1xi1>, %on_true : tensor<256x1xi64>, %on_false : tensor<256x1xi64>) -> tensor<256x1xi64> {
  %0 = "stablehlo.select"(%pred, %on_true, %on_false) : (tensor<256x1xi1>, tensor<256x1xi64>, tensor<256x1xi64>) -> tensor<256x1xi64>
  func.return %0 : tensor<256x1xi64>
}
