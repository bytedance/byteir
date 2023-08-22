func.func @gemm_rrr_f32(%arg0 : tensor<256x256xf32>, %arg1 : tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}
