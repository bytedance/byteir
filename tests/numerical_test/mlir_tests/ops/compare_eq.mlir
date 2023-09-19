func.func @compare_eq(%arg0 : tensor<1x256x1024xi64>, %arg1 : tensor<1x256x1024xi64>) -> tensor<1x256x1024xi1> {
  %0 = mhlo.compare  EQ, %arg0, %arg1,  SIGNED : (tensor<1x256x1024xi64>, tensor<1x256x1024xi64>) -> tensor<1x256x1024xi1>
  return %0 : tensor<1x256x1024xi1>
}
