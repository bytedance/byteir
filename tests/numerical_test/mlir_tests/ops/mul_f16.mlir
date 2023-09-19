func.func @multiply(%arg0 : tensor<1x32x256x128xf16>, %arg1 : tensor<1x32x256x128xf16>) -> tensor<1x32x256x128xf16> {
  %0 = mhlo.multiply %arg0, %arg1 : tensor<1x32x256x128xf16>
  return %0 : tensor<1x32x256x128xf16>
}
