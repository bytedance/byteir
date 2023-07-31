func.func @gemm_rrr_f16(%arg0 : tensor<1024x4096xf16>, %arg1 : tensor<4096x4096xf16>) -> tensor<1024x4096xf16> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<1024x4096xf16>, tensor<4096x4096xf16>) -> tensor<1024x4096xf16>
  return %0 : tensor<1024x4096xf16>
}
