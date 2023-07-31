func.func @concat(%arg0 : tensor<1x32x256x64xf16>, %arg1 : tensor<1x32x256x64xf16>) -> tensor<1x32x256x128xf16> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 3 : i64} : (tensor<1x32x256x64xf16>, tensor<1x32x256x64xf16>) -> tensor<1x32x256x128xf16>
  return %0 : tensor<1x32x256x128xf16>
}
