func.func @power(%arg0 : tensor<1x256x4096xf32>) -> tensor<1x256x4096xf32> {
  %cst = mhlo.constant dense<3.000000e+00> : tensor<1x256x4096xf32>
  %0 = mhlo.power %arg0, %cst : tensor<1x256x4096xf32>
  return %0 : tensor<1x256x4096xf32>
}

