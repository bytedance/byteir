func.func @log_plus_one(%arg0 : tensor<256x1xf16>) -> tensor<256x1xf16> { 
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<256x1xf16>) -> tensor<256x1xf16>
  func.return %0 : tensor<256x1xf16>
}
