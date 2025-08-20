func.func @byteir.l2_norm(%arg0: tensor<10x128xf32>) -> tensor<10x128xf32> {
  %0 = stablehlo.custom_call @byteir.l2_norm(%arg0) {byteir_attrs = {axis = [1], eps_outside_sqrt = true, epsilon = 9.9999999999999998E-13 : f64}} : (tensor<10x128xf32>) -> tensor<10x128xf32>
  return %0 : tensor<10x128xf32>
}
