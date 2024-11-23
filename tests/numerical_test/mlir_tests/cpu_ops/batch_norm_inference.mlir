module {
  func.func @main(%arg0: tensor<10x32x10xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32xf32>, %arg4: tensor<32xf32>) -> tensor<10x32x10xf32> {
    %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<10x32x10xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<10x32x10xf32>
    return %0 : tensor<10x32x10xf32>
  }
}
