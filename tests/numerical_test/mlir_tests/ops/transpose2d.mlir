func.func @transpose2d(%arg0 : tensor<4096x13696xf16>) -> tensor<13696x4096xf16> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4096x13696xf16>) -> tensor<13696x4096xf16>
  return %0 : tensor<13696x4096xf16>
}