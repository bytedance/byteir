func.func @reduce_f32(%input : tensor<256x5xf32>) -> tensor<256xf32> {
  %0 = stablehlo.constant  dense<-0.000000e+00> : tensor<f32>
  %1 = "stablehlo.reduce"(%input, %0) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %2 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    dimensions = array<i64: 1>
  } : (tensor<256x5xf32>, tensor<f32>) -> tensor<256xf32>
  func.return %1 : tensor<256xf32>
}
