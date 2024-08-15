func.func @forward(%arg0: tensor<6x8x5xf32>, %arg1: tensor<6x1x5xf32>) -> tensor<6x8x5xf32> {
    %c = stablehlo.constant dense<0> : tensor<1x1xi64>
    %0 = "stablehlo.scatter"(%arg0, %c, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      stablehlo.return %arg3 : tensor<f32>
    }) : (tensor<6x8x5xf32>, tensor<1x1xi64>, tensor<6x1x5xf32>) -> tensor<6x8x5xf32>
    return %0 : tensor<6x8x5xf32>
}