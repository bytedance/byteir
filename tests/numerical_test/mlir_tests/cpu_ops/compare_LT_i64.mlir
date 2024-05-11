func.func @compare_LT_i64(%arg0 : tensor<256x1xi64>, %arg1 : tensor<256x1xi64>) -> tensor<256x1xi1> { 
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction LT>,
    compare_type = #stablehlo<comparison_type SIGNED>
  } : (tensor<256x1xi64>, tensor<256x1xi64>) -> tensor<256x1xi1>
  func.return %0 : tensor<256x1xi1>
}
