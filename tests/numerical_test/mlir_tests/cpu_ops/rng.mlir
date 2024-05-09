func.func @rng_f16() -> tensor<256x120xf16> { 
  %0 = stablehlo.constant dense<[256, 120]> : tensor<2xi64>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
  %3 = "stablehlo.rng"(%2, %1, %0) {
    rng_distribution = #stablehlo<rng_distribution UNIFORM>,
    device = "host"
  } : (tensor<f16>, tensor<f16>, tensor<2xi64>) -> tensor<256x120xf16>
  func.return %3 : tensor<256x120xf16>
}
