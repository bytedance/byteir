func.func @gemm_rrr_f32(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
