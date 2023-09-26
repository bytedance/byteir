func.func @concat2(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<2xi64>) {
  %0 = mhlo.reshape %arg0 : (tensor<i64>) -> tensor<1xi64>
  %1 = mhlo.reshape %arg1 : (tensor<i64>) -> tensor<1xi64>
  %2 = "mhlo.concatenate"(%0, %1) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  return %2 : tensor<2xi64>
}
