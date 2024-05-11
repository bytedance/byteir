func.func @compare_LT_f32(%arg0 : tensor<256x1xf32>, %arg1 : tensor<256x1xf32>) -> tensor<256x1xi1> { 
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction LT>,
    compare_type = #stablehlo<comparison_type FLOAT>
  } : (tensor<256x1xf32>, tensor<256x1xf32>) -> tensor<256x1xi1>
  func.return %0 : tensor<256x1xi1>
}
