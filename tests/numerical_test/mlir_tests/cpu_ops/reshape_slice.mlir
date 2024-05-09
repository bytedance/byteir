func.func @reshape_slice_reshape(%arg0: tensor<256x2xf16>) -> (tensor<256xf16>) {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<256x2xf16>) -> tensor<256x1x2xf16>
  %1 = "stablehlo.slice"(%0) {limit_indices = array<i64: 256, 1, 1>, start_indices = array<i64: 0, 0, 0>, strides =array<i64: 1, 1, 1>} : (tensor<256x1x2xf16>) -> tensor<256x1x1xf16>
  %2 = "stablehlo.reshape"(%1) : (tensor<256x1x1xf16>) -> tensor<256xf16>
  func.return %2 : tensor<256xf16>
}
