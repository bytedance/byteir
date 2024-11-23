module {
  func.func @main(%arg0: tensor<10x32x10xf16>, %arg1: tensor<32xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>, %arg4: tensor<32xf16>) -> tensor<10x32x10xf16> {
    %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<10x32x10xf16>, tensor<32xf16>, tensor<32xf16>, tensor<32xf16>, tensor<32xf16>) -> tensor<10x32x10xf16>
    return %0 : tensor<10x32x10xf16>
  }
}
