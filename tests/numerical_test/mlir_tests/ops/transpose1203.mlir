func.func @transpose1203(%arg0 : tensor<12x3x7x6xf16>) -> tensor<3x7x12x6xf16> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 2, 0, 3]> : tensor<4xi64>} : (tensor<12x3x7x6xf16>) -> tensor<3x7x12x6xf16>
  return %0 : tensor<3x7x12x6xf16>
}
