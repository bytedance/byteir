func.func @divide_f16(%arg0 : tensor<256x1xf16>, %arg1 : tensor<256x1xf16>) -> tensor<256x1xf16> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<256x1xf16>
  func.return %0 : tensor<256x1xf16>
}
