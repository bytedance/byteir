func.func @divide(%arg0 : tensor<1x256x4096xf32>) -> tensor<1x256x4096xf32> {
  %cst = mhlo.constant dense<4.096000e+03> : tensor<1x256x4096xf32>
  %0 = mhlo.divide %arg0, %cst : tensor<1x256x4096xf32>
  return %0 : tensor<1x256x4096xf32>
}
