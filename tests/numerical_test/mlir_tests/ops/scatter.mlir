
func.func @scatter(%arg0 : tensor<256x1xi64>, %arg1 : tensor<256x4096xf32>) -> tensor<32000x4096xf32> {
    %cst = mhlo.constant dense<0.000000e+00> : tensor<32000x4096xf32>
    %0 = "mhlo.scatter"(%cst, %arg0, %arg1) ({
    ^bb0(%arg66: tensor<f32>, %arg67: tensor<f32>):
      %395 = mhlo.add %arg66, %arg67 : tensor<f32>
      mhlo.return %395 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<32000x4096xf32>, tensor<256x1xi64>, tensor<256x4096xf32>) -> tensor<32000x4096xf32>
    return %0 : tensor<32000x4096xf32>
}
