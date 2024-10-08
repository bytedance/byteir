func.func @byteir.softmax(%arg0: tensor<10x128xf32>) -> tensor<10x128xf32> {
  %0 = stablehlo.custom_call @byteir.softmax(%arg0) {byteir_attrs = {axis = 1 : i64}} : (tensor<10x128xf32>) -> tensor<10x128xf32>
  return %0 : tensor<10x128xf32>
}
