func.func @broadcast_in_dim(%arg0 : tensor<256x1xi64>) -> tensor<256x200xi64> { 
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<256x1xi64>) -> tensor<256x200xi64>
  func.return %0 : tensor<256x200xi64>
}
