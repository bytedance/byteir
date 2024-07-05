func.func @broadcast_in_dim(%arg0 : tensor<128x1xi64>) -> tensor<128x3xi64> { 
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1>
  } : (tensor<128x1xi64>) -> tensor<128x3xi64>
  func.return %0 : tensor<128x3xi64>
}
