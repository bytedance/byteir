func.func @select_f32(%pred : tensor<256x1xi1>, %on_true : tensor<256x1xf32>, %on_false : tensor<256x1xf32>) -> tensor<256x1xf32> {
  %0 = "stablehlo.select"(%pred, %on_true, %on_false) : (tensor<256x1xi1>, tensor<256x1xf32>, tensor<256x1xf32>) -> tensor<256x1xf32>
  func.return %0 : tensor<256x1xf32>
}
