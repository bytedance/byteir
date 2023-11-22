func.func @add(%arg0 : tensor<256x256xf16>, %arg1 : tensor<256x256xf16>) -> tensor<256x256xf16> {
  %0 = mhlo.add %arg0, %arg1 : tensor<256x256xf16>
  return %0 : tensor<256x256xf16>
}
