func.func @logistic(%arg0 : tensor<1x256x1024xf16>) -> tensor<1x256x1024xf16> {
  %0 = mhlo.logistic %arg0 : tensor<1x256x1024xf16>
  return %0 : tensor<1x256x1024xf16>
}
