func.func @slice(%arg0 : tensor<1x1x1024x1024xi1>) -> tensor<1x1x256x1024xi1> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[1, 1, 256, 1024]> : tensor<4xi64>, start_indices = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<4xi64>} : (tensor<1x1x1024x1024xi1>) -> tensor<1x1x256x1024xi1>
  return %0 : tensor<1x1x256x1024xi1>
}
