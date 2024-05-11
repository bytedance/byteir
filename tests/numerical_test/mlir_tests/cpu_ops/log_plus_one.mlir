func.func @log_plus_one(%arg0 : tensor<256x64xf16>) -> tensor<256x64xf16> { 
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<256x64xf16>) -> tensor<256x64xf16>
  func.return %0 : tensor<256x64xf16>
}
