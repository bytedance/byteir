func.func @subtract_f16(%arg0 : tensor<256x200xf16>, %arg1 : tensor<256x200xf16>) -> tensor<256x200xf16> { 
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<256x200xf16>, tensor<256x200xf16>) -> (tensor<256x200xf16>)
  func.return %0 : tensor<256x200xf16>
}
