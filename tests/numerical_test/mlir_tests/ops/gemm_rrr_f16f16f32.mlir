func.func @main(%arg0: tensor<256x128xf16>, %arg1: tensor<128x256xf16>) -> tensor<256x256xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1): (tensor<256x128xf16>, tensor<128x256xf16>)-> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}
