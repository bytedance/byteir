func.func @byteir.arg_min$return_2(%arg0: tensor<3x128xf32>) -> (tensor<3xf32>, tensor<3xi64>) {
  %0:2 = stablehlo.custom_call @byteir.arg_min(%arg0) {byteir_attrs = {axis = 1 : i64, keep_dims = false, select_last_index = false}} : (tensor<3x128xf32>) -> (tensor<3xf32>, tensor<3xi64>)
  return %0#0, %0#1 : tensor<3xf32>, tensor<3xi64>
}
