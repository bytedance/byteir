module {
  func.func @main(%arg0: tensor<3x2xf32>) -> tensor<3x8xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
    %1 = stablehlo.constant() {value = dense<1.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %2 = stablehlo.constant() {value = dense<2.0> : tensor<8x16xf32>} : () -> tensor<8x16xf32>
    %3 = stablehlo.constant() {value = dense<3.0> : tensor<16xf32>} : () -> tensor<16xf32>
    %4 = stablehlo.constant() {value = dense<4.0> : tensor<16x8xf32>} : () -> tensor<16x8xf32>
    %5 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    %6 = ccl.all_gather %5 {axis = 0 : i64, replica_groups = [[0, 1, 2, 3]], synchronous = true} : (tensor<2x3xf32>) -> tensor<8x3xf32>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<8x3xf32>) -> tensor<3x8xf32>
    %8 = stablehlo.transpose %4, dims = [1, 0] : (tensor<16x8xf32>) -> tensor<8x16xf32>
    %9 = stablehlo.dot %7, %8 : (tensor<3x8xf32>, tensor<8x16xf32>) -> tensor<3x16xf32>
    %10 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<16xf32>
    %12 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<16xf32>) -> tensor<16xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<16xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [1] : (tensor<16xf32>) -> tensor<3x16xf32>
    %15 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<3x16xf32>) -> tensor<3x16xf32>
    %16 = stablehlo.add %14, %15 : tensor<3x16xf32>
    %17 = stablehlo.transpose %2, dims = [1, 0] : (tensor<8x16xf32>) -> tensor<16x8xf32>
    %18 = stablehlo.dot %16, %17 : (tensor<3x16xf32>, tensor<16x8xf32>) -> tensor<3x8xf32>
    %19 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %20 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<8xf32>) -> tensor<8xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<8xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<8xf32>) -> tensor<3x8xf32>
    %23 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<3x8xf32>) -> tensor<3x8xf32>
    %24 = stablehlo.add %22, %23 : tensor<3x8xf32>
    %25 = ccl.all_reduce %24 {reduction = "sum", replica_groups = [[0, 1, 2, 3]], synchronous = true} : (tensor<3x8xf32>) -> tensor<3x8xf32>
    return %25 : tensor<3x8xf32>
  }
}

