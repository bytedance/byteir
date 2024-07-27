func.func @byteir.addn(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %0 = stablehlo.custom_call @byteir.addn(%arg0, %arg1, %arg2) {byteir_attrs = {}} : (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}
